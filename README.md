# HR Policy RAG Chatbot using Langchain

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
about HR Policy documents using OpenAI and Pinecone.

## Tech Stack
- Python
- LangChain
- OpenAI (text-embedding-3-small, gpt-3.5-turbo)
- Pinecone (vector database)
- Streamlit (UI)

## How it works
1. PDF is loaded and split into chunks
2. Chunks are embedded using OpenAI embeddings
3. Vectors are stored in Pinecone
4. User query is embedded and matched against stored vectors
5. Top matching chunks are passed to GPT as context
6. GPT answers based only on the retrieved context

## Note
The `resources/` folder is not included in this repo.
Create a `resources/` folder and add your own HR Policy PDF,
then update the file path in `dataprocessor.py` and `app.py`.

## Setup
1. Clone the repo
2. Install dependencies: pip install -r requirements.txt
3. Create a .env file:
OPENAI_API_KEY=your_key_here
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=your_index_name

4. Run dataprocessor.py once to load PDF into Pinecone
5. Launch the app: streamlit run app.py

