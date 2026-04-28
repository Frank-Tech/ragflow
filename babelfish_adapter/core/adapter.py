# ═══════════════════════════════════════════════════════════════════════════════
# BABELFISH ADAPTER — ragflow
# ═══════════════════════════════════════════════════════════════════════════════
#
# This file implements the 3 functions that lexus-test calls to run external
# flows through the babelfish proxy:
#
#   run(...)           — execute a Canvas, yield event-stream steps
#   list_payloads()    — return all testable payload names
#   list_flow_groups() — return canvas/subflow metadata for trace mapping
#
# STRUCTURE (mirrors agentic-soc-platform/babelfish_adapter/core/adapter.py):
#
#   1. BABELFISH BOILERPLATE (KEEP AS-IS)
#      Generic run wrapper that handles context setup, Langfuse callbacks,
#      trace metadata, and cleanup. Stays in lockstep with the ASP version —
#      do not diverge unless ASP gets the same change.
#
#   2. PROJECT-SPECIFIC (RAGFLOW)
#      Canvas registry, payload definitions, flow groups, input builders, and
#      execute_flow() that drives ragflow's Canvas runtime.
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import uuid
from typing import AsyncGenerator, Dict, List

from babelfish_adapter.core.context import babelfish_context, _current_session_id, _current_flow_id


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: BABELFISH BOILERPLATE — KEEP AS-IS                            ║
# ║                                                                         ║
# ║  Generic wrapper copied from agentic-soc-platform.                      ║
# ║  Handles: context setup, Langfuse callbacks, trace metadata, cleanup.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _build_langfuse_callbacks(trace_id: str) -> tuple[list, object | None]:
    """Build Langfuse callbacks for the main flow's trace."""
    pub = os.environ.get("CLIENT_LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("CLIENT_LANGFUSE_SECRET_KEY")
    host = os.environ.get("CLIENT_LANGFUSE_HOST")
    if not (pub and sec and host):
        return [], None
    from langfuse import Langfuse
    from langfuse.langchain import CallbackHandler

    Langfuse(public_key=pub, secret_key=sec, base_url=host)
    handler = CallbackHandler(public_key=pub, trace_context={"trace_id": trace_id})
    return [handler], handler


async def run(
    *,
    mode: str,
    flow_id: str,
    payload_name: str,
    subflow_server_ids: dict | None = None,
    model: str | None = None,
) -> AsyncGenerator[dict, None]:
    """Execute a flow and yield event-stream steps.                  KEEP AS-IS

    This is the generic run wrapper. It:
      1. Validates mode
      2. Mints the parent flow's session_id (UUID4) and Langfuse trace_id
      3. Builds Langfuse callbacks for the parent flow
      4. Sets babelfish_context (mode / flow_id / subflow bookkeeping only —
         session_id is NOT in the context; every flow role mints its own and
         passes it explicitly to LLMAPI.get_model)
      5. Calls YOUR execute_flow() to do the actual work
      6. Collects per-invocation subflow trace records
      7. Yields __trace_metadata__ with the parent session_id + subflows
      8. Cleans up (flushes Langfuse, resets context)

    The only project-specific part is execute_flow() — see PART 2 below.

    lexus-test no longer pre-mints a session_id and passes it in; the adapter
    owns minting and reports every session back via ``__trace_metadata__``.

    ``model`` kwarg (A-design, added for lexus-test integration):
        Optional OpenAI model name to pin for this invocation (e.g.
        ``"gpt-4o-2024-08-06"``). Lexus-test passes this from its own
        settings so baseline and babelfish runs always resolve to the same
        snapshot. When ``None`` (the standalone case), the LLM factory
        falls back to its pre-existing model source — so callers that
        don't pass ``model`` continue to work unchanged.
    """
    if mode not in ("baseline", "babelfish"):
        raise ValueError(f"Invalid mode: {mode}. Expected 'baseline' or 'babelfish'.")

    # Parent flow's identity — minted here, reported back via __trace_metadata__.
    parent_session_id = str(uuid.uuid4())
    parent_client_trace_id = str(uuid.uuid4()).replace("-", "")

    callbacks, main_handler = _build_langfuse_callbacks(parent_client_trace_id)

    subflow_invocations: list = []
    token = babelfish_context.set(
        {
            "mode": mode,
            "flow_id": flow_id,
            "subflow_server_ids": subflow_server_ids or {},
            "subflow_invocations": subflow_invocations,
            "model_override": model,
        }
    )

    session_token = _current_session_id.set(parent_session_id)
    flow_id_token = _current_flow_id.set(flow_id)
    try:
        # ── This is the only line that calls ragflow's code ──
        async for step in execute_flow(
            payload_name=payload_name,
            session_id=parent_session_id,
            callbacks=callbacks,
        ):
            yield step

        actual_trace_id = (
            main_handler.last_trace_id if main_handler and main_handler.last_trace_id
            else parent_client_trace_id
        )

        yield {
            "__trace_metadata__": {
                "session_id": parent_session_id,
                "client_trace_id": actual_trace_id,
                "server_trace_id": parent_session_id.replace("-", ""),
                "subflow_invocations": subflow_invocations,
            }
        }
    finally:
        for cb in callbacks:
            if hasattr(cb, "_langfuse_client"):
                try:
                    cb._langfuse_client.flush()
                except Exception:
                    pass
        try:
            _current_session_id.reset(session_token)
        except ValueError:
            _current_session_id.set(None)
        try:
            _current_flow_id.reset(flow_id_token)
        except ValueError:
            _current_flow_id.set(None)
        try:
            babelfish_context.reset(token)
        except ValueError:
            babelfish_context.set(None)


# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 2: PROJECT-SPECIFIC — RAGFLOW                                    ║
# ║                                                                         ║
# ║  Canvas registry, payloads, flow groups, execute_flow().                ║
# ║  Filled in after the canvas event-shape spike confirms how to drive     ║
# ║  ragflow's Canvas runtime in-process and what shape its events take.    ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


# TODO(spike): populated by inspect_canvas_events.py output. See:
#   scripts/external_flows/ragflow/inspect_canvas_events.py
#
# Will define:
#   _CANVASES        — entry_id → Canvas DSL JSON path
#   _PAYLOADS        — entry_id → list of payload names
#   _FLOW_GROUPS     — flow/subflow metadata for trace mapping
#   execute_flow()   — async generator that drives Canvas in-process
#   list_payloads()  — cross product of canvases × payloads
#   list_flow_groups() — walks each canvas DSL to extract subflow system prompts


async def execute_flow(
    *,
    payload_name: str,
    session_id: str,
    callbacks: list,
) -> AsyncGenerator[dict, None]:
    """Run a ragflow Canvas and yield its event stream.

    Implementation deferred until canvas spike completes — we need to know
    the actual event dict shape Canvas produces before writing the translator.
    """
    raise NotImplementedError(
        "execute_flow is pending the canvas event-shape spike. "
        "Run scripts/external_flows/ragflow/inspect_canvas_events.py first."
    )
    # Unreachable — present so this stays a valid async generator function.
    if False:
        yield {}


def list_payloads() -> List[str]:
    """Return all testable payload names.

    Format: "entry_id:payload_name". Implementation deferred — see PART 2 TODO.
    """
    raise NotImplementedError("list_payloads pending PART 2 implementation")


def list_flow_groups() -> List[Dict]:
    """Return flow/subflow metadata for trace mapping.

    Walks each canvas DSL to extract the parent flow's system prompt and any
    nested-agent (subflow) system prompts. Implementation deferred — see PART 2.
    """
    raise NotImplementedError("list_flow_groups pending PART 2 implementation")
