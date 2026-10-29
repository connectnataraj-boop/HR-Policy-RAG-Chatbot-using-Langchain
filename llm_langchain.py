import os
from dotenv import load_dotenv
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
import cohere

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


def store_in_pinecone(chunks, namespace):
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY")
    )
    vectorstore = PineconeVectorStore.from_texts(
        texts=chunks,
        embedding=embeddings,
        index_name=os.getenv("PINECONE_INDEX_NAME"),
        namespace=namespace,
        ids=[f"{namespace}-{i}" for i in range(len(chunks))],
    )
    return vectorstore

# Rerank


def rerank(query, docs, top_n=4):
    """Re-order retrieved chunks by relevance. Falls back safely if it fails."""
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        return docs[:top_n]
    try:
        co = cohere.ClientV2(api_key=api_key)
        res = co.rerank(
            model=os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5"),
            query=query,
            documents=[d.page_content for d in docs],
            top_n=top_n,
        )
        return [docs[r.index] for r in res.results]
    except Exception as e:
        print(f"Rerank failed, using vector order: {e}")
        return docs[:top_n]

# Run


def query_processor(query: str, vectorstore, use_rerank=True,
                    fetch_k=20, top_n=4) -> str:
    if use_rerank:
        candidates = vectorstore.similarity_search(query, k=fetch_k)
        results = rerank(query, candidates, top_n=top_n)
    else:
        results = vectorstore.similarity_search(query, k=top_n)

    context = "\n\n".join([doc.page_content for doc in results])

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.4,
        api_key=os.getenv("OPENAI_API_KEY")
    )
    messages = [
        {"role": "system",
         "content": "Answer based only on the context provided. "
                    "If the answer is not in the context, say you don't know."},
        {"role": "user", "content": f"Query: {query}\n\nContext:\n{context}"}
    ]
    response = llm.invoke(messages)
    return response.content
