"""MCP server for the HR Policy chatbot.

Put this file in the same folder as app.py and llm_langchain.py.
It exposes three read-only tools that any MCP-capable AI app can use:
  - get_leave_policy   : fixed facts for one leave type (no AI, no cost)
  - search_hr_policy   : retrieve + rerank passages from the HR policy PDF
  - ask_hr_policy      : answer a question from the HR policy (your full RAG)

Run the test:   python test_mcp_client.py
(The client starts this server itself. Startup takes 10-20 seconds
because the server connects to Pinecone before it accepts calls.)
"""
import asyncio
import hashlib
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("hr-policy")

BASE_DIR = Path(__file__).resolve().parent
POLICY_PDF = BASE_DIR / "resources" / "HRPolicy.pdf"
MAX_QUERY_CHARS = 500
FETCH_K = 20            # chunks retrieved before reranking
MAX_TOP_N = 8           # most chunks a caller can ask for
TOOL_TIMEOUT = 90       # seconds before a RAG tool gives up

# Facts taken from HRPolicy.pdf. Update this if the policy changes.
LEAVE_POLICY = {
    "annual": (
        "Annual (earned) leave: 24 days per year, credited monthly. "
        "Up to 15 unused days can be carried forward. "
        "Encashment of earned leave is permitted only on separation."
    ),
    "casual": (
        "Casual leave is for short-term personal needs. "
        "The policy does not state a number of days."
    ),
    "sick": (
        "Sick leave covers absences due to illness. Medical certificates are "
        "expected for extended periods. The policy does not state a number of days."
    ),
    "compensatory": (
        "Compensatory off may be taken for work done on weekends or holidays, "
        "by prior agreement and with manager approval. It must be used within "
        "thirty days of accrual."
    ),
}

ALIASES = {
    "earned": "annual", "earned leave": "annual", "annual leave": "annual",
    "casual leave": "casual", "sick leave": "sick",
    "comp off": "compensatory", "comp-off": "compensatory",
    "compensatory off": "compensatory", "compoff": "compensatory",
}

_state = {}


def log(message: str) -> None:
    """Log to stderr. Never print to stdout: stdout carries the MCP messages."""
    print(message, file=sys.stderr, flush=True)


def _normalise(leave_type: str) -> str:
    key = " ".join(leave_type.lower().split())
    return ALIASES.get(key, key)


def _get_vectorstore():
    """Connect to the existing Pinecone data. Index the PDF only if it is missing."""
    if "vs" in _state:
        return _state["vs"], _state["doc_id"]

    from llm_langchain import load_pdf, get_chunks, store_in_pinecone
    from langchain_openai import OpenAIEmbeddings
    from langchain_pinecone import PineconeVectorStore

    doc_id = hashlib.md5(POLICY_PDF.read_bytes()).hexdigest()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    vs = PineconeVectorStore(
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        embedding=embeddings,
        namespace=doc_id,
    )
    if not vs.similarity_search("leave", k=1):
        log("No vectors found for this document, indexing it now...")
        vs = store_in_pinecone(get_chunks(load_pdf(str(POLICY_PDF))), doc_id)

    _state.update(vs=vs, doc_id=doc_id)
    return vs, doc_id


def _check_text(text: str, label: str):
    """Return an error message if the input is empty or too long, else None."""
    if not text or not text.strip():
        return f"Please provide a {label}."
    if len(text) > MAX_QUERY_CHARS:
        return f"The {label} is too long (maximum {MAX_QUERY_CHARS} characters)."
    return None


# ---------- The slow work runs in a worker thread (see the tools below) ----------
def _search_sync(query: str, top_n: int) -> str:
    from llm_langchain import rerank
    vs, _ = _get_vectorstore()
    candidates = vs.similarity_search(query, k=FETCH_K)
    docs = rerank(query, candidates, top_n=top_n)
    if not docs:
        return "No relevant passages found."
    return "\n\n".join(f"[{i}] {d.page_content}" for i, d in enumerate(docs, 1))


def _ask_sync(question: str) -> str:
    from llm_langchain import query_processor
    vs, doc_id = _get_vectorstore()
    answer, _stats = query_processor(
        question, vs, doc_id=doc_id,
        use_rerank=True, use_cache=True, fetch_k=FETCH_K, top_n=4,
    )
    return answer


async def _run_blocking(func, *args) -> str:
    """Run slow, blocking code in a thread so the server stays responsive."""
    return await asyncio.wait_for(asyncio.to_thread(func, *args), timeout=TOOL_TIMEOUT)


# ---------- Tools ----------
@mcp.tool()
def get_leave_policy(leave_type: str) -> str:
    """Return the HR policy summary for one leave type.

    Supported types: annual (also 'earned'), casual, sick, compensatory.
    """
    problem = _check_text(leave_type, "leave type")
    if problem:
        return problem
    key = _normalise(leave_type)
    if key in LEAVE_POLICY:
        return LEAVE_POLICY[key]
    return f"Not found. Supported leave types: {', '.join(LEAVE_POLICY)}."


@mcp.tool()
async def search_hr_policy(query: str, top_n: int = 4) -> str:
    """Search the HR policy document and return the most relevant passages.

    Read-only. Retrieves 20 passages, reranks them, and returns the best top_n.
    """
    problem = _check_text(query, "query")
    if problem:
        return problem
    top_n = max(1, min(int(top_n), MAX_TOP_N))
    try:
        return await _run_blocking(_search_sync, query.strip(), top_n)
    except asyncio.TimeoutError:
        log("search_hr_policy timed out")
        return "The policy search timed out. Please try again."
    except Exception as exc:  # fail safe: never crash the server
        log(f"search_hr_policy failed: {exc!r}")
        return "The policy search is temporarily unavailable."


@mcp.tool()
async def ask_hr_policy(question: str) -> str:
    """Answer a question using only the HR policy document (RAG with reranking).

    If the policy does not contain the answer, the reply says so.
    """
    problem = _check_text(question, "question")
    if problem:
        return problem
    try:
        return await _run_blocking(_ask_sync, question.strip())
    except asyncio.TimeoutError:
        log("ask_hr_policy timed out")
        return "The HR policy assistant timed out. Please try again."
    except Exception as exc:  # fail safe: never crash the server
        log(f"ask_hr_policy failed: {exc!r}")
        return "The HR policy assistant is temporarily unavailable."


if __name__ == "__main__":
    # Connect to Pinecone BEFORE accepting calls, outside the event loop.
    # This is the same path as the direct test that already worked.
    try:
        log("Connecting to the HR policy index (10-20 seconds)...")
        _get_vectorstore()
        log("Index ready.")
    except Exception as exc:
        log(f"Warm-up failed: {exc!r}. The leave lookup still works; "
            "the two RAG tools will report an error.")
    mcp.run()  # stdio transport (local use)
