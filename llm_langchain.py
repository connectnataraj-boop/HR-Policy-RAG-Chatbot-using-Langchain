import os
import time
from dotenv import load_dotenv
import cohere
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# ---------- Settings ----------
# Price per 1 million tokens (USD). Copy the gpt-4o-mini numbers from OpenAI's
PRICE_IN_PER_1M = float(os.getenv("PRICE_IN_PER_1M", "0"))
PRICE_OUT_PER_1M = float(os.getenv("PRICE_OUT_PER_1M", "0"))

MAX_ANSWER_TOKENS = 300      # caps output length (less time, less cost)
MAX_CACHE_ENTRIES = 200

SYSTEM_PROMPT = (
    "Answer based only on the context provided. "
    "Be concise. "
    "If the answer is not in the context, say you don't know."
)

# ---------- Clients (created once, reused) ----------
_llm = None
_cohere = None
_answer_cache = {}


def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.4,
            max_tokens=MAX_ANSWER_TOKENS,
            api_key=os.getenv("OPENAI_API_KEY"),
        )
    return _llm


def get_cohere():
    global _cohere
    if _cohere is None:
        _cohere = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
    return _cohere


# ---------- Ingestion ----------
def load_pdf(path):
    """path can be a file path or a file-like object (for example BytesIO)."""
    reader = PdfReader(path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    return splitter.split_text(text)


def store_in_pinecone(chunks, namespace):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    return PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=namespace,
        ids=[f"{namespace}-{i}" for i in range(len(chunks))],
    )


# ---------- Reranking ----------
def rerank(query, docs, top_n=4):
    """Re-order retrieved chunks by relevance. Falls back to vector order on failure."""
    if len(docs) <= top_n:
        return docs
    if not os.getenv("COHERE_API_KEY"):
        return docs[:top_n]
    try:
        res = get_cohere().rerank(
            model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_n,
        )
        return [docs[r.index] for r in res.results]
    except Exception as e:
        print(f"Rerank failed, using vector order: {e}")
        return docs[:top_n]


# ---------- Query ----------
def query_processor(query: str, vectorstore, doc_id: str = "",
                    use_rerank: bool = True, use_cache: bool = True,
                    fetch_k: int = 20, top_n: int = 4):
    """Returns (answer, stats). stats has tokens, cost and the time of each stage."""
    start = time.perf_counter()
    empty = {"tokens_in": 0, "tokens_out": 0, "cost": 0.0,
             "t_search": 0.0, "t_rerank": 0.0, "t_llm": 0.0}

    key = (doc_id, " ".join(query.lower().split()), use_rerank, fetch_k, top_n)
    if use_cache and key in _answer_cache:
        stats = dict(empty, cached=True, latency=time.perf_counter() - start)
        return _answer_cache[key], stats

    # 1. Retrieve
    t0 = time.perf_counter()
    k = fetch_k if use_rerank else top_n
    candidates = vectorstore.similarity_search(query, k=k)
    t_search = time.perf_counter() - t0

    # 2. Rerank (optional)
    t0 = time.perf_counter()
    results = rerank(query, candidates, top_n=top_n) if use_rerank else candidates
    t_rerank = time.perf_counter() - t0

    # 3. Generate
    context = "\n\n".join(doc.page_content for doc in results)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"},
    ]
    t0 = time.perf_counter()
    response = get_llm().invoke(messages)
    t_llm = time.perf_counter() - t0

    usage = response.usage_metadata or {}
    tokens_in = usage.get("input_tokens", 0)
    tokens_out = usage.get("output_tokens", 0)
    cost = (tokens_in / 1e6) * PRICE_IN_PER_1M + (tokens_out / 1e6) * PRICE_OUT_PER_1M

    answer = response.content
    if use_cache:
        if len(_answer_cache) >= MAX_CACHE_ENTRIES:
            _answer_cache.pop(next(iter(_answer_cache)))
        _answer_cache[key] = answer

    stats = {
        "tokens_in": tokens_in, "tokens_out": tokens_out, "cost": cost,
        "t_search": t_search, "t_rerank": t_rerank, "t_llm": t_llm,
        "latency": time.perf_counter() - start, "cached": False,
    }
    return answer, stats
