# ═══════════════════════════════════════════════════════════════════════════════
# BABELFISH CONTEXT — Generic Protocol Layer
# ═══════════════════════════════════════════════════════════════════════════════
#
# This module is PROJECT-INDEPENDENT. Copy it as-is into any project that
# integrates with lexus-test via the babelfish proxy.
#
# ─── Design: explicit session_id, no inheritance ────────────────────────────
#
# Every flow role (parent flow, every subflow) is responsible for minting its
# own fresh UUID4 ``session_id`` and passing it explicitly to the LLM factory
# (``LLMAPI.get_model(..., session_id=...)``).  There is NO implicit default,
# no ContextVar-based inheritance, and no parent-session override dance.  This
# makes the "forgot to wrap a concurrent Send fan-out" bug structurally
# impossible — every LLM call site must declare which session it belongs to,
# or the call refuses to build.
#
# ``babelfish_context`` still exists, but it only carries cross-cutting state
# that's invariant across all calls inside one adapter.run() invocation:
#   - mode                    ("babelfish" | "baseline")
#   - flow_id                 (X-Flow-ID header)
#   - subflow_server_ids      (which system messages belong to tracked subflows)
#   - subflow_invocations     (accumulator for __trace_metadata__)
#   - tracked_message_hashes  (SHA-256 of every babelfish-managed system message;
#                              the routing wrapper checks this set at call time)
#
# ─── Headers are resolved at call time, not construction time ───────────────
#
# X-Session-ID, X-Flow-ID and X-Api-Key are read from the ContextVars below
# inside ``_RoutingCompletions.create`` (see this module). There is no need to
# create chat clients fresh per invocation — cached clients pick up the right
# values automatically because the headers are built per-call from contextvars.
#
# ═══════════════════════════════════════════════════════════════════════════════

import asyncio
import contextvars
import hashlib
import json
import os
import uuid as _uuid
from typing import Optional, TypedDict

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    # Ragflow doesn't use LangChain — the isolation handler is only needed
    # for LangGraph-based projects (ASP). Provide a plain object fallback.
    BaseCallbackHandler = object  # type: ignore[misc,assignment]


class _CallbackIsolationHandler(BaseCallbackHandler):
    """No-op handler that prevents LangChain parent-callback inheritance.

    LangChain's ``CallbackManager.configure()`` treats ``callbacks=[]``
    (empty list) as falsy and falls through to the parent graph's ambient
    callback context — so any Langfuse handler from the parent leaks into
    the child graph and records duplicate observations on the wrong trace.

    Passing a non-empty list (even with a no-op handler) forces LangChain
    to create a fresh ``CallbackManager`` with only our handlers, blocking
    inheritance from the parent.

    In ragflow (which doesn't use LangChain), this class is never used —
    the Canvas runtime has no callback inheritance to block.
    """


# ── ContextVar Definition ─────────────────────────────────────────────────────

class BabelfishContextData(TypedDict):
    mode: str                       # "baseline" | "babelfish"
    flow_id: str                    # X-Flow-ID header for babelfish routing
    subflow_server_ids: dict        # system_message_content → {"msg_hash": ..., "flow_uuid": ...} (tracked subflows)
    subflow_invocations: list       # accumulator: list[{msg_hash, client_trace_id, server_session_id}]
    tracked_message_hashes: set     # SHA-256 hashes of every babelfish-managed system message (parent + subflows).
                                    # Routing wrapper at chat-completions.create time checks this set to decide
                                    # babelfish vs direct OpenAI. Mutated by execute_flow() after DSL load.


babelfish_context: contextvars.ContextVar[Optional[BabelfishContextData]] = contextvars.ContextVar(
    "babelfish_context", default=None
)


# ── Ragflow-specific: session_id / flow_id bridges ──────────────────────────
# ASP passes session_id and flow_id explicitly via
# LLMAPI.get_model(session_id=..., flow_id=...).
# Ragflow's LLM factory (Base.__init__ in chat_model.py) has no such
# parameters — it's called by ragflow's component system. These ContextVars
# bridge the gap: adapter.run() sets them before execute_flow(), and
# Base.__init__ reads them for the X-Session-ID / X-Flow-ID headers.
# Subflows override _current_flow_id with their own flow_uuid so holy-grail
# deduplicates shared subflows into one babel_fish_agentic_flows row.

_current_session_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_session_id", default=None
)

_current_flow_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_flow_id", default=None
)

_current_client_trace_id: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "_current_client_trace_id", default=None
)


# ── Subflow Helpers ───────────────────────────────────────────────────────────
# Subflow identification is still ContextVar-based because ``subflow_server_ids``
# is cross-cutting (the same map applies to every LLM call inside one
# adapter.run()).  Session identity is NOT in the ContextVar — the helper
# below mints fresh UUIDs at every invocation and returns them to the caller,
# who passes them explicitly to ``LLMAPI.get_model()``.

def _is_tracked_subflow(system_message_content: str) -> dict | None:
    """Return ``{"msg_hash": ..., "flow_uuid": ...}`` if tracked subflow, else None."""
    ctx = babelfish_context.get()
    if not ctx:
        return None
    server_ids = ctx.get("subflow_server_ids", {})
    return server_ids.get(system_message_content)


def _build_subflow_callback(client_trace_id: str) -> list:
    """Build a fresh Langfuse CallbackHandler for a single subflow invocation."""
    pub = os.environ.get("CLIENT_LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("CLIENT_LANGFUSE_SECRET_KEY")
    host = os.environ.get("CLIENT_LANGFUSE_HOST")
    if not (pub and sec and host):
        return []
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
        Langfuse(public_key=pub, secret_key=sec, base_url=host)
        handler = CallbackHandler(public_key=pub, trace_context={"trace_id": client_trace_id})
        return [handler]
    except Exception:
        return []


def flush_callbacks(callbacks: list) -> None:
    """Flush all Langfuse clients to ensure trace data is sent."""
    for cb in callbacks:
        if hasattr(cb, "_langfuse_client"):
            try:
                cb._langfuse_client.flush()
            except Exception:
                pass


# ── Canonical system-message hash ─────────────────────────────────────────────
# Anchored on babelfish's hash_system_message
# (tai_nexus_holy_grail/babelfish/utils/nexus_util.py:29) — that's the function
# that compares slot ownership in Redis. Lexus-test's copy is byte-identical
# today; if either drifts, this is the one we MUST follow.
#
# Canonical input shape: {"role": "system", "content": <str>}. Strip everything
# else (e.g. ``name`` on tool-result-bearing messages) before hashing so the
# call-time hash matches the registration-time hash.

def hash_system_message_content(content: str) -> str:
    """SHA-256 of the canonical system-message dict for a given content string."""
    canonical = {"role": "system", "content": content}
    return hashlib.sha256(
        json.dumps(canonical, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()


def _is_tracked_system_content(content: str) -> bool:
    """True if the system-message content is in the tracked-message registry."""
    ctx = babelfish_context.get()
    if not ctx:
        return False
    return hash_system_message_content(content) in ctx.get("tracked_message_hashes", set())


# ── Langfuse trace_id auto-injection ──────────────────────────────────────────
# When the langfuse.openai drop-in wrapper is active, every call creates its
# own Langfuse trace unless ``trace_id`` is passed in. This helper rebinds
# ``client.chat.completions.create`` so it reads the current trace_id from a
# ContextVar at call time — pinning every call inside one adapter.run() to one
# client trace, which lexus-test polls.

def _inject_langfuse_trace_id(client, trace_id_var):
    orig_create = client.chat.completions.create
    if asyncio.iscoroutinefunction(orig_create):
        async def _traced(*args, **kwargs):
            tid = trace_id_var.get()
            if tid and "trace_id" not in kwargs:
                kwargs["trace_id"] = tid
            return await orig_create(*args, **kwargs)
    else:
        def _traced(*args, **kwargs):
            tid = trace_id_var.get()
            if tid and "trace_id" not in kwargs:
                kwargs["trace_id"] = tid
            return orig_create(*args, **kwargs)
    client.chat.completions.create = _traced


# ── Per-call routing client ───────────────────────────────────────────────────
# At chat.completions.create time, decide babelfish vs direct OpenAI based on
# whether the call's system message is in the tracked-message registry. Headers
# for the babelfish path are read from ContextVars at CALL time (not at client
# construction), so the cached-stale-header bug is structurally impossible.

class _RoutingCompletions:
    def __init__(self, direct_client, bf_client, nexus_api_key):
        self._direct = direct_client
        self._bf = bf_client
        self._nexus_api_key = nexus_api_key

    def create(self, *args, **kwargs):
        messages = kwargs.get("messages") or (args[0] if args else [])
        sm = next(
            (m.get("content") for m in messages if isinstance(m, dict) and m.get("role") == "system"),
            None,
        )
        if sm and _is_tracked_system_content(sm):
            sid = _current_session_id.get()
            fid = _current_flow_id.get()
            if not sid or not fid:
                raise RuntimeError(
                    "babelfish-tracked LLM call missing X-Session-ID or X-Flow-ID — "
                    "_current_session_id/_current_flow_id contextvar is None at call time. "
                    "The adapter must set both before any tracked LLM call."
                )
            extra_headers = dict(kwargs.pop("extra_headers", None) or {})
            extra_headers.update({
                "X-Session-ID": sid,
                "X-Flow-ID": fid,
                "X-Api-Key": self._nexus_api_key,
                "X-Auto-Approve": "true",
            })
            return self._bf.chat.completions.create(*args, extra_headers=extra_headers, **kwargs)
        return self._direct.chat.completions.create(*args, **kwargs)


class _RoutingChat:
    def __init__(self, direct_client, bf_client, nexus_api_key):
        self.completions = _RoutingCompletions(direct_client, bf_client, nexus_api_key)


class _RoutingClient:
    """Proxy that exposes ``.chat.completions.create`` and forwards everything
    else to the direct (provider) client. Built once per ``Base.__init__``;
    routing decision happens per call."""

    def __init__(self, direct_client, bf_client, nexus_api_key):
        self._direct = direct_client
        self._bf = bf_client
        self.chat = _RoutingChat(direct_client, bf_client, nexus_api_key)

    def __getattr__(self, name):
        return getattr(self._direct, name)


def build_chat_clients(*, api_key: str, provider_base_url: str, timeout: int):
    """Called by ragflow ``rag/llm/chat_model.py:Base.__init__``.

    Returns a ``(sync_client, async_client)`` pair when the babelfish adapter
    is active in this context; returns ``None`` when inactive (caller falls
    back to plain ``OpenAI`` / ``AsyncOpenAI``).

    Both clients in the pair are routing wrappers: per-call, they inspect the
    request's system message and dispatch to either the babelfish proxy (with
    contextvar-derived headers built at call time) or the provider's API
    directly. Both underlying paths are wrapped with ``langfuse.openai`` so
    Langfuse client tracing captures every call regardless of routing.

    Known limitation — sub-Agent X-Flow-ID:
        ``agent/component/agent_with_tools.py`` mints a fresh session_id for
        sub-Agent invocations but discards the flow_id returned by
        ``mint_flow_session`` (it sets ``_current_session_id`` but not
        ``_current_flow_id``). Today ragflow declares zero subflows in
        ``list_flow_groups``, so sub-Agent SMs are never in the registry and
        their calls go direct to OpenAI — flow_id doesn't matter. If subflow
        tracking is ever enabled for ragflow, the agent_with_tools mint block
        must also override ``_current_flow_id`` with the subflow's flow_uuid,
        otherwise this wrapper will send the parent's X-Flow-ID for subflow
        calls.
    """
    ctx = babelfish_context.get()
    if not ctx or ctx.get("mode") != "babelfish":
        return None

    bf_base_url = os.environ["OPENAI_BASE_URL"]
    nexus_api_key = os.environ["NEXUS_API_KEY"]

    try:
        from langfuse.openai import OpenAI as _OAI, AsyncOpenAI as _AOAI
        has_langfuse = True
    except ImportError:
        from openai import OpenAI as _OAI, AsyncOpenAI as _AOAI
        has_langfuse = False

    direct_sync = _OAI(api_key=api_key, base_url=provider_base_url, timeout=timeout)
    bf_sync     = _OAI(api_key=api_key, base_url=bf_base_url,       timeout=timeout)
    direct_async = _AOAI(api_key=api_key, base_url=provider_base_url, timeout=timeout)
    bf_async     = _AOAI(api_key=api_key, base_url=bf_base_url,       timeout=timeout)

    if has_langfuse:
        for c in (direct_sync, bf_sync, direct_async, bf_async):
            _inject_langfuse_trace_id(c, _current_client_trace_id)

    return (
        _RoutingClient(direct_sync, bf_sync, nexus_api_key),
        _RoutingClient(direct_async, bf_async, nexus_api_key),
    )


def mint_flow_session(system_message_content: str) -> tuple[str, list, str | None]:
    """Mint a fresh ``session_id`` for one flow role invocation.

    Called once per flow entry point: top-level ``adapter.run()`` mints one
    for the parent flow, and every sub-agent / node function mints its own
    when it starts making LLM calls.  The returned ``session_id`` is a fresh
    UUID4 that the caller MUST pass to every ``LLMAPI.get_model`` call within
    its scope — there is no ContextVar fallback.

    If the given system_message_content is registered as a tracked subflow in
    ``babelfish_context["subflow_server_ids"]``, the call is recorded in
    ``subflow_invocations`` (for ``__trace_metadata__``) and Langfuse callbacks
    scoped to this invocation are returned.  Otherwise the call is considered
    parent-flow work — no tracked row is recorded, and the returned callbacks
    list is empty (the parent's own callbacks come from ``adapter.run()``).

    Returns:
        (session_id, callbacks, flow_id) — the caller passes ``session_id``
        and ``flow_id`` to every ``LLMAPI.get_model`` call in its scope,
        and threads ``callbacks`` into the RunnableConfig for graph
        invocations so Langfuse traces land on the right client trace id.
        ``flow_id`` is the subflow's own ``flow_uuid`` for tracked subflows,
        or the parent's ``flow_id`` for parent-flow work (always the value
        the caller should send as ``X-Flow-ID``). It is ``None`` only when
        running outside the adapter, where it is not used by the LLM factory.
    """
    session_id = str(_uuid.uuid4())

    parent_ctx = babelfish_context.get()
    if not parent_ctx:
        # Running outside the adapter (tests, CLI). Caller still gets a
        # session_id so the LLM call site has one to pass.
        # Return isolation handler to prevent parent-callback leakage.
        return session_id, [_CallbackIsolationHandler()], None

    subflow_info = _is_tracked_subflow(system_message_content)
    if not subflow_info:
        # Parent flow work — the adapter's own callbacks already trace this.
        # Return isolation handler to prevent parent-callback leakage.
        return session_id, [_CallbackIsolationHandler()], parent_ctx["flow_id"]

    # Tracked subflow: both keys are required by contract; missing keys
    # surface as KeyError rather than silently falling back.
    msg_hash = subflow_info["msg_hash"]
    flow_uuid = subflow_info["flow_uuid"]

    client_trace_id = str(_uuid.uuid4()).replace("-", "")
    parent_ctx["subflow_invocations"].append({
        "msg_hash": msg_hash,
        "client_trace_id": client_trace_id,
        "server_session_id": session_id,
    })
    callbacks = _build_subflow_callback(client_trace_id)
    return session_id, callbacks, flow_uuid
