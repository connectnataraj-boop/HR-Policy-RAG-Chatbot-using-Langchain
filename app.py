import streamlit as st
from llm_langchain import load_pdf, get_chunks, store_in_pinecone, query_processor

st.title("HR Policy Chatbot")
st.write("Ask anything about the HR Policy document.")


@st.cache_resource
def setup():
    text = load_pdf("resources/HRPolicy.pdf")
    chunks = get_chunks(text)
    vectorstore = store_in_pinecone(chunks)
    return vectorstore


vectorstore = setup()
if "messages" not in st.session_state:
    st.session_state.messages = []


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])


query = st.chat_input("Ask your question...")

if query:
    # Show user message
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append({"role": "user", "content": query})

    # Get answer
    with st.spinner("Thinking..."):
        answer = query_processor(query)

    # Show assistant answer
    with st.chat_message("assistant"):
        st.write(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
