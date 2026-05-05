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
# ─── ADAPTER YIELD CONTRACT ───────────────────────────────────────────────────
#
# The run() async generator must yield exactly two special dicts (in order)
# after the flow execution completes. The lexus-test runner reads these and
# ignores all other yielded steps (which are passed through as-is).
#
# 1. __tool_calls__ — adapter-parsed tool call data
#    The adapter owns event parsing (LangGraph messages, Canvas events, etc.)
#    and yields the extracted data in this standardized shape:
#
#    {
#        "__tool_calls__": {
#            "tool_call_groups": list[list[tuple[str, str]]],
#                # Ordered list of tool-call groups. Each group is one
#                # AIMessage's tool_calls: [(call_id, tool_name), ...].
#                # Parallel calls within one AIMessage are one group.
#            "tool_outputs": dict[str, Any],
#                # Mapping of call_id → tool output content.
#                # Only successful (non-errored) outputs are used downstream.
#            "errored_call_ids": list[str],
#                # call_ids whose ToolMessage had status="error".
#                # These are stripped from the "clean" tool path.
#        }
#    }
#
#    If the adapter cannot extract tool calls from its event stream (e.g.
#    ragflow Canvas events don't contain them), yield empty values.
#    The pipeline will extract tool data from Langfuse traces instead.
#
# 2. __trace_metadata__ — identity and trace IDs for polling
#
#    {
#        "__trace_metadata__": {
#            "session_id": str,            # REQUIRED — parent flow's UUID4
#            "client_trace_id": str,       # REQUIRED — Langfuse client trace ID
#            "server_trace_id": str,       # REQUIRED — session_id without hyphens
#            "subflow_invocations": list,  # [{msg_hash, client_trace_id,
#                                          #   server_session_id}, ...]
#        }
#    }
#
#    The runner raises RuntimeError if __trace_metadata__ is missing or
#    lacks any of the three required keys.
#
# ═══════════════════════════════════════════════════════════════════════════════

import os
import uuid
from typing import AsyncGenerator, Dict, List

from babelfish_ragflow_adapter.core.context import (
    babelfish_context, _current_session_id, _current_flow_id, _current_client_trace_id,
)

# ╔═══════════════════════════════════════════════════════════════════════════╗
# ║  PART 1: BABELFISH BOILERPLATE — KEEP AS-IS                            ║
# ║                                                                         ║
# ║  Generic wrapper copied from agentic-soc-platform.                      ║
# ║  Handles: context setup, Langfuse callbacks, trace metadata, cleanup.   ║
# ╚═══════════════════════════════════════════════════════════════════════════╝


def _build_langfuse_callbacks(trace_id: str) -> tuple[list, object | None]:
    """Build Langfuse callbacks for the main flow's trace.

    For ragflow, the LangChain CallbackHandler is not available (no langchain
    installed). Client tracing is handled via langfuse.openai's drop-in
    AsyncOpenAI wrapper in LiteLLMBase._construct_completion_args().
    This function returns empty when langchain isn't present.
    """
    pub = os.environ.get("CLIENT_LANGFUSE_PUBLIC_KEY")
    sec = os.environ.get("CLIENT_LANGFUSE_SECRET_KEY")
    host = os.environ.get("CLIENT_LANGFUSE_HOST")
    if not (pub and sec and host):
        return [], None
    try:
        from langfuse import Langfuse
        from langfuse.langchain import CallbackHandler
        Langfuse(public_key=pub, secret_key=sec, base_url=host)
        handler = CallbackHandler(public_key=pub, trace_context={"trace_id": trace_id})
        return [handler], handler
    except (ImportError, ModuleNotFoundError):
        return [], None


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
    trace_id_token = _current_client_trace_id.set(parent_client_trace_id)

    # ── Langfuse client trace env bridging ──────────────────────────────
    # langfuse.openai reads LANGFUSE_PUBLIC_KEY / SECRET_KEY / HOST from
    # env. Our keys use the CLIENT_LANGFUSE_* prefix. Bridge them.
    _langfuse_env_backup: dict[str, str | None] = {}
    _lf_env_map = {
        "LANGFUSE_PUBLIC_KEY": "CLIENT_LANGFUSE_PUBLIC_KEY",
        "LANGFUSE_SECRET_KEY": "CLIENT_LANGFUSE_SECRET_KEY",
        "LANGFUSE_HOST": "CLIENT_LANGFUSE_HOST",
    }
    for lf_key, client_key in _lf_env_map.items():
        _langfuse_env_backup[lf_key] = os.environ.get(lf_key)
        client_val = os.environ.get(client_key)
        if client_val:
            os.environ[lf_key] = client_val
    # ──────────────────────────────────────────────────────────────────

    try:
        # ── This is the only line that calls ragflow's code ──
        async for step in execute_flow(
            payload_name=payload_name,
            session_id=parent_session_id,
            callbacks=callbacks,
        ):
            yield step

        # Canvas events don't contain LangGraph-style tool calls.
        # Tool calls are captured from Langfuse traces by the pipeline.
        yield {
            "__tool_calls__": {
                "tool_call_groups": [],
                "tool_outputs": {},
                "errored_call_ids": [],
            }
        }

        yield {
            "__trace_metadata__": {
                "session_id": parent_session_id,
                "client_trace_id": parent_client_trace_id,
                "server_trace_id": parent_session_id.replace("-", ""),
                "subflow_invocations": subflow_invocations,
            }
        }
    finally:
        # Restore Langfuse env vars
        for lf_key, prev_val in _langfuse_env_backup.items():
            if prev_val is None:
                os.environ.pop(lf_key, None)
            else:
                os.environ[lf_key] = prev_val
        for cb in callbacks:
            if hasattr(cb, "_langfuse_client"):
                try:
                    cb._langfuse_client.flush()
                except Exception:
                    pass
        # Flush the langfuse.openai singleton client (used by the drop-in
        # AsyncOpenAI wrapper in chat_model.py). Without this, observations
        # buffered on the background queue may not reach Langfuse before
        # lexus-test starts polling, producing "trace not found" / "0
        # observations" failures. Use get_client() to grab the singleton
        # rather than constructing a new Langfuse() (which would have an
        # empty queue).
        try:
            from langfuse import get_client
            get_client().flush()
        except Exception:
            pass
        try:
            _current_client_trace_id.reset(trace_id_token)
        except ValueError:
            _current_client_trace_id.set(None)
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
# ║  Drives ragflow's Canvas runtime in-process using templates from        ║
# ║  agent/templates/ with patched LLM IDs, tool API keys, and KB IDs.     ║
# ╚═══════════════════════════════════════════════════════════════════════════╝

import json
import pathlib

# ── Template registry ─────────────────────────────────────────────────────────

_TEMPLATES: Dict[str, str] = {
    # Only templates with at least one Agent that passes the Agent Inclusion
    # Contract (see tai-nexus-lexus-test/src/.../core/external_adapter.py).
    # Each of these contributes 1 FLOW (a multi-tool Agent with chained
    # sub-Agents-as-tools, triggered from the graph entry).
    "deep_research": "agent/templates/deep_research.json",
    "seo_article_writer": "agent/templates/seo_article_writer.json",
}

def _load_payloads() -> Dict[str, list[str]]:
    """Load test payloads from payloads.json (20 queries per template)."""
    payload_path = pathlib.Path(__file__).parent.parent / "payloads.json"
    with payload_path.open() as f:
        return json.load(f)


_PAYLOADS: Dict[str, list[str]] | None = None


def _get_payloads() -> Dict[str, list[str]]:
    global _PAYLOADS
    if _PAYLOADS is None:
        _PAYLOADS = _load_payloads()
    return _PAYLOADS

# Tools to strip from web_search_assistant (require paid SerpApi key)
_PAID_TOOLS = {"Google", "Bing", "SerpApi"}


def _get_ragflow_repo() -> pathlib.Path:
    """Resolve the ragflow repo path from env or default."""
    return pathlib.Path(
        os.environ.get("RAGFLOW_REPO", pathlib.Path.home() / "PycharmProjects" / "ragflow")
    ).resolve()


def _load_and_patch_dsl(entry_id: str) -> str:
    """Load a template DSL, apply runtime patches, return JSON string.

    Patches applied:
    - llm_id override from RAGFLOW_LLM_ID
    - Tavily API key injection from RAGFLOW_TAVILY_API_KEY
    - KB ID injection from RAGFLOW_KB_FAQ / RAGFLOW_KB_RESEARCH
    - Strip paid search tools from web_search_assistant
    """
    ragflow_repo = _get_ragflow_repo()
    template_path = ragflow_repo / _TEMPLATES[entry_id]
    with template_path.open() as f:
        data = json.load(f)

    # Unwrap template wrapper (DSL is nested under "dsl" key)
    if "dsl" in data and "components" in data.get("dsl", {}):
        data = data["dsl"]

    if "components" not in data:
        raise RuntimeError(f"Template {entry_id} has no 'components' key in DSL")

    llm_override = os.environ.get("RAGFLOW_LLM_ID")
    tavily_key = os.environ.get("RAGFLOW_TAVILY_API_KEY")
    kb_faq = os.environ.get("RAGFLOW_KB_FAQ")
    kb_research = os.environ.get("RAGFLOW_KB_RESEARCH")

    for cpn_id, cpn in data["components"].items():
        obj = cpn.get("obj", {})
        params = obj.get("params", {})

        # Override llm_id
        if llm_override and "llm_id" in params:
            params["llm_id"] = llm_override

        # Patch tools
        tools = params.get("tools", [])
        patched_tools = []
        for tool in tools:
            tn = tool.get("component_name", "")
            tp = tool.get("params", {})

            # Strip paid search tools from web_search_assistant
            if entry_id == "web_search_assistant" and tn in _PAID_TOOLS:
                continue

            # Inject Tavily API key
            if tn in ("TavilySearch", "TavilyExtract") and tavily_key:
                tp["api_key"] = tavily_key

            # Override llm_id in nested Agent tools
            if tn == "Agent" and llm_override and "llm_id" in tp:
                tp["llm_id"] = llm_override

            # Inject Tavily key in nested Agent tools' inner tools
            for inner_tool in tp.get("tools", []):
                itn = inner_tool.get("component_name", "")
                itp = inner_tool.get("params", {})
                if itn in ("TavilySearch", "TavilyExtract") and tavily_key:
                    itp["api_key"] = tavily_key

            patched_tools.append(tool)
        params["tools"] = patched_tools

        # Inject KB IDs for Retrieval tools
        for tool in params.get("tools", []):
            if tool.get("component_name") == "Retrieval":
                tp = tool.get("params", {})
                if not tp.get("dataset_ids") and not tp.get("kb_ids"):
                    if entry_id == "reflective_academic_paper_generator":
                        tp["dataset_ids"] = [kb_research] if kb_research else []
                    else:
                        tp["dataset_ids"] = [kb_faq] if kb_faq else []

    return json.dumps(data)


_settings_initialized = False


def _ensure_ragflow_settings():
    """Initialize ragflow settings once (loads service_conf.yaml, llm_factories, etc.)."""
    global _settings_initialized
    if _settings_initialized:
        return
    import sys
    ragflow_repo = _get_ragflow_repo()
    if str(ragflow_repo) not in sys.path:
        sys.path.insert(0, str(ragflow_repo))
    # Generate service_conf.yaml from RAGFLOW_* env vars so ragflow's
    # settings.init_settings() connects to the correct services (Docker
    # service names from .env.lexus, not localhost defaults).
    conf_dir = ragflow_repo / "conf"
    conf_dir.mkdir(exist_ok=True)
    _write_service_conf(conf_dir / "service_conf.yaml")
    os.environ.setdefault("RAGFLOW_CONF_DIR", str(conf_dir))
    from common import settings
    settings.init_settings()
    _seed_llm_for_tools()
    _settings_initialized = True


def _seed_llm_for_tools():
    """Upsert the RAGFLOW_LLM_ID row in ragflow's `llm` table with
    is_tools=True, so LLMBundle.bind_tools actually binds the tools.

    Why: ragflow's `llm` table is seeded from conf/llm_factories.json
    which carries stable names like 'gpt-4o', not date-pinned snapshots
    like 'gpt-4o-2024-08-06'. Lexus-test pins the snapshot for baseline
    parity, so the snapshot name misses the JSON seed → LLMService.query
    returns no row → is_tools defaults to False → LLMBundle.bind_tools
    drops the tools binding silently → orchestrator has no tool-call
    surface for babelfish to learn from. Upserting here at adapter init
    closes that gap without modifying upstream ragflow code.
    """
    llm_id = os.environ.get("RAGFLOW_LLM_ID")
    if not llm_id or "@" not in llm_id:
        return
    llm_name, _, fid = llm_id.partition("@")
    from api.db.services.llm_service import LLMService
    existing = LLMService.query(llm_name=llm_name, fid=fid)
    if existing:
        row = existing[0]
        if not row.is_tools:
            LLMService.update_by_id(row.id, {"is_tools": True})
        return
    LLMService.save(
        llm_name=llm_name,
        fid=fid,
        model_type="chat",
        tags="LLM,CHAT,128K",
        max_tokens=128000,
        is_tools=True,
    )


def _write_service_conf(path: pathlib.Path) -> None:
    """Write a service_conf.yaml from RAGFLOW_* env vars."""
    mysql_host = os.environ.get("RAGFLOW_MYSQL_HOST", "localhost")
    mysql_port = os.environ.get("RAGFLOW_MYSQL_PORT", "3306")
    mysql_password = os.environ.get("RAGFLOW_MYSQL_PASSWORD", "infini_rag_flow")
    es_host = os.environ.get("RAGFLOW_ES_HOST", "localhost")
    es_port = os.environ.get("RAGFLOW_ES_PORT", "9200")
    es_password = os.environ.get("RAGFLOW_ELASTIC_PASSWORD", "infini_rag_flow")
    minio_host = os.environ.get("RAGFLOW_MINIO_HOST", "localhost")
    minio_port = os.environ.get("RAGFLOW_MINIO_PORT", "9000")
    minio_user = os.environ.get("RAGFLOW_MINIO_USER", "rag_flow")
    minio_password = os.environ.get("RAGFLOW_MINIO_PASSWORD", "infini_rag_flow")
    redis_host = os.environ.get("RAGFLOW_REDIS_HOST", "localhost")
    redis_port = os.environ.get("RAGFLOW_REDIS_PORT", "6379")
    redis_password = os.environ.get("RAGFLOW_REDIS_PASSWORD", "infini_rag_flow")

    conf = f"""ragflow:
  host: 0.0.0.0
  http_port: 9380
mysql:
  name: 'rag_flow'
  user: 'root'
  password: '{mysql_password}'
  host: '{mysql_host}'
  port: {mysql_port}
  max_connections: 100
  stale_timeout: 300
minio:
  user: '{minio_user}'
  password: '{minio_password}'
  host: '{minio_host}:{minio_port}'
es:
  hosts: 'http://{es_host}:{es_port}'
  username: 'elastic'
  password: '{es_password}'
redis:
  db: 1
  username: ''
  password: '{redis_password}'
  host: '{redis_host}:{redis_port}'
"""
    path.write_text(conf)


# ── Contract functions ────────────────────────────────────────────────────────


async def execute_flow(
    *,
    payload_name: str,
    session_id: str,
    callbacks: list,
) -> AsyncGenerator[dict, None]:
    """Run a ragflow Canvas and yield its event stream.

    Canvas events are yielded as-is — lexus-test extracts tool calls from
    Langfuse traces (via langfuse.openai wrapper on the AsyncOpenAI client
    in LiteLLMBase._construct_completion_args), not from the event stream.
    The event stream just needs to flow without errors.
    """
    entry_id, _, payload_index = payload_name.partition(":")
    if entry_id not in _TEMPLATES:
        raise RuntimeError(f"Unknown entry_id: {entry_id}. Available: {list(_TEMPLATES.keys())}")

    payloads = _get_payloads()
    entry_payloads = payloads.get(entry_id, [])
    try:
        idx = int(payload_index)
    except (ValueError, TypeError):
        idx = 0
    if idx < 0 or idx >= len(entry_payloads):
        raise RuntimeError(
            f"Payload index {idx} out of range for {entry_id} "
            f"(has {len(entry_payloads)} payloads)"
        )
    query = entry_payloads[idx]

    _ensure_ragflow_settings()

    dsl = _load_and_patch_dsl(entry_id)
    tenant_id = os.environ.get("RAGFLOW_TENANT_ID", "adapter-tenant")

    from agent.canvas import Canvas
    canvas = Canvas(dsl=dsl, tenant_id=tenant_id)

    async for event in canvas.run(query=query):
        yield event


def list_payloads() -> List[str]:
    """Return all testable payload names.

    Format: "entry_id:index" — 20 payloads per template (180 total).
    """
    payloads = _get_payloads()
    result = []
    for entry_id in _TEMPLATES:
        entry_payloads = payloads.get(entry_id, [])
        for i in range(len(entry_payloads)):
            result.append(f"{entry_id}:{i}")
    return result


def list_flow_groups() -> List[Dict]:
    """Return flow/subflow metadata for trace mapping.

    Reads sys_prompt directly from template DSL JSON — no Canvas
    instantiation, no DB connections, no settings init needed.
    """
    groups = []
    for entry_id in _TEMPLATES:
        dsl_str = _load_and_patch_dsl(entry_id)
        dsl = json.loads(dsl_str)
        components = dsl.get("components", {})

        # Find the first Agent/LLM component's sys_prompt
        flow_system_message = ""
        for cpn_id, cpn in components.items():
            params = cpn.get("obj", {}).get("params", {})
            sys_prompt = params.get("sys_prompt", "")
            if sys_prompt:
                flow_system_message = sys_prompt
                break

        if not flow_system_message:
            flow_system_message = f"ragflow:{entry_id}"

        groups.append({
            "entry_id": entry_id,
            "flow": {
                "name": entry_id,
                "system_message": flow_system_message,
            },
            "subflows": [],
        })

    return groups
