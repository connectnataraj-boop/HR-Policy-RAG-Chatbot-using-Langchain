import io
import hashlib
import streamlit as st
from llm_langchain import load_pdf, get_chunks, store_in_pinecone, query_processor

st.title("HR Policy Chatbot")
st.write("Upload an HR policy PDF, then ask questions about it.")

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

vectorstore = setup(uploaded.getvalue())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

query = st.chat_input("Ask your question...")

if query:
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    with st.spinner("Thinking..."):
        answer = query_processor(query, vectorstore)

    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
