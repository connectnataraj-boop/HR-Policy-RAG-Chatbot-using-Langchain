import io
import hashlib
import streamlit as st
from llm_langchain import load_pdf, get_chunks, store_in_pinecone, query_processor

st.title("HR Policy Chatbot")
st.write("Upload an HR policy PDF, then ask questions about it.")

# ---------- Sidebar controls ----------
st.sidebar.header("Settings")
use_rerank = st.sidebar.checkbox("Use reranking", value=True)
use_cache = st.sidebar.checkbox("Use cache", value=True)
fetch_k = st.sidebar.slider("Chunks retrieved (before rerank)", 4, 30, 20)
top_n = st.sidebar.slider("Chunks sent to the LLM", 2, 6, 4)

if "totals" not in st.session_state:
    st.session_state.totals = {"questions": 0, "cache_hits": 0,
                               "tokens": 0, "cost": 0.0, "time": 0.0}
if "messages" not in st.session_state:
    st.session_state.messages = []

if st.sidebar.button("Clear chat and totals"):
    st.session_state.messages = []
    st.session_state.totals = {"questions": 0, "cache_hits": 0,
                               "tokens": 0, "cost": 0.0, "time": 0.0}

uploaded = st.file_uploader("Upload a PDF", type="pdf")


@st.cache_resource(show_spinner="Reading and indexing the document...")
def setup(file_bytes: bytes):
    namespace = hashlib.md5(file_bytes).hexdigest()
    text = load_pdf(io.BytesIO(file_bytes))
    chunks = get_chunks(text)
    return store_in_pinecone(chunks, namespace)


if uploaded is None:
    st.info("Please upload a PDF to start.")
    st.stop()

file_bytes = uploaded.getvalue()
doc_id = hashlib.md5(file_bytes).hexdigest()
vectorstore = setup(file_bytes)

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("caption"):
            st.caption(msg["caption"])

query = st.chat_input("Ask your question...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        answer, s = query_processor(
            query, vectorstore, doc_id=doc_id, use_rerank=use_rerank,
            use_cache=use_cache, fetch_k=fetch_k, top_n=top_n)

    if s["cached"]:
        caption = f"From cache | {s['latency']:.2f}s | cost $0"
    else:
        caption = (
            f"{s['tokens_in']} in / {s['tokens_out']} out tokens | "
            f"${s['cost']:.6f} | total {s['latency']:.1f}s "
            f"(search {s['t_search']:.2f}s, rerank {s['t_rerank']:.2f}s, "
            f"LLM {s['t_llm']:.2f}s)")

    with st.chat_message("assistant"):
        st.write(answer)
        st.caption(caption)
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "caption": caption})

    t = st.session_state.totals
    t["questions"] += 1
    t["cache_hits"] += int(s["cached"])
    t["tokens"] += s["tokens_in"] + s["tokens_out"]
    t["cost"] += s["cost"]
    t["time"] += s["latency"]

# ---------- Running totals ----------
t = st.session_state.totals
st.sidebar.markdown("---")
st.sidebar.subheader("Totals this session")
st.sidebar.write(f"Questions: {t['questions']}")
st.sidebar.write(f"Cache hits: {t['cache_hits']}")
st.sidebar.write(f"Total tokens: {t['tokens']}")
st.sidebar.write(f"Total cost: ${t['cost']:.5f}")
if t["questions"]:
    st.sidebar.write(f"Average time: {t['time'] / t['questions']:.2f}s")
