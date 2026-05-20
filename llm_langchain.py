import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

load_dotenv()

# Load PDF


def load_pdf(path):
    reader = PdfReader(path)
    pages = [page.extract_text() for page in reader.pages]
    return "\n".join(pages)

# Split into chunks


def get_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150
    )
    return splitter.split_text(text)

# Embed and store in Pinecone


def store_in_pinecone(chunks):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME")
    )
    return vectorstore


# Run
def query_processor(query: str, vectorstore) -> str:
    results = vectorstore.similarity_search(query, k=4)
    context = "\n\n".join([doc.page_content for doc in results])

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.4,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    messages = [
        {"role": "system", "content": "Answer based only on the context provided."},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
    ]
    response = llm.invoke(messages)
    return response.content


if __name__ == "__main__":
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        query_processor(query)
