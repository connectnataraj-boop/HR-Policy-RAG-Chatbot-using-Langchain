# HR Policy RAG Chatbot — Built With LangChain

> **RAG · LangChain · OpenAI · Pinecone · Streamlit · Production Framework**

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about HR Policy documents — built using **LangChain**, the production-standard framework used in real GenAI engineering roles.

> 🔗 Also see the from-scratch version: [HR-Policy-RAG-Chatbot-without-Langchain](https://github.com/connectnataraj-boop/HR-Policy-RAG-Chatbot-without-Langchain)

---

## 📌 Table of Contents

- [Why I Built Two Versions](#-why-i-built-two-versions)
- [Architecture](#-architecture)
- [LangChain vs From Scratch — What Each Abstracts](#-langchain-vs-from-scratch--what-each-abstracts)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [How It Works](#-how-it-works)
- [Sample Questions & Answers](#-sample-questions--answers)
- [Key Learnings — Using LangChain for RAG](#-key-learnings--using-langchain-for-rag)
- [Setup Instructions](#-setup-instructions)
- [Author](#-author)

---

## 🤔 Why I Built Two Versions

Most people learn RAG one of two ways:

- **Only LangChain** → fast to build, but don't understand what's happening inside
- **Only from scratch** → deep understanding, but not how production teams actually work

I built **both** — in the right order:

| Version | Approach | What I Learned |
|---|---|---|
| [Without LangChain](https://github.com/connectnataraj-boop/HR-Policy-RAG-Chatbot-without-Langchain) | Every component handwritten | How RAG actually works under the hood |
| **This repo** (With LangChain) | LangChain abstractions | How production GenAI teams actually build it |

Building the from-scratch version first meant I arrived at this repo already knowing what `RecursiveCharacterTextSplitter`, `PineconeVectorStore`, and `similarity_search` are actually doing. That knowledge makes me a better engineer when using LangChain — I can debug failures, tune parameters, and reason about tradeoffs instead of just calling functions blindly.

> *"Understanding what LangChain replaces makes you far better at using LangChain."*

---

## 🏗️ Architecture

### Indexing Pipeline (Run Once — loads document into Pinecone)

```
HR Policy PDF
      │
      ▼
┌──────────────────────────────────────┐
│  PdfReader (pypdf)                   │  extracts raw text from all pages
│  load_pdf()                          │
└──────────────────┬───────────────────┘
                   │  full document text
                   ▼
┌──────────────────────────────────────┐
│  RecursiveCharacterTextSplitter      │  chunk_size=500, chunk_overlap=150
│  get_chunks()                        │  splits on paragraphs → sentences → words
└──────────────────┬───────────────────┘
                   │  list of text chunks
                   ▼
┌──────────────────────────────────────┐
│  OpenAIEmbeddings                    │  text-embedding-3-small
│  store_in_pinecone()                 │  1536-dim vector per chunk
│  PineconeVectorStore.from_texts()    │  upserts all vectors + metadata
└──────────────────────────────────────┘
                   │
           Pinecone Index  ← ready for querying
```

### Query Pipeline (Every User Question)

```
User Question  →  Streamlit UI  (app.py)
                        │
                        ▼
          ┌─────────────────────────┐
          │  vectorstore            │  embeds query → cosine similarity search
          │  .similarity_search()   │  returns top-4 matching chunks (k=4)
          └────────────┬────────────┘
                       │  list of Document objects (page_content + metadata)
                       ▼
          ┌─────────────────────────┐
          │  ChatOpenAI             │  gpt-3.5-turbo, temperature=0.4
          │  query_processor()      │  system: "Answer only from context"
          │                         │  user: query + retrieved context
          └────────────┬────────────┘
                       │  grounded answer string
                       ▼
              Streamlit UI  ←  displays answer to user
```

---

## 🔄 LangChain vs From Scratch — What Each Abstracts

This table shows exactly what LangChain replaces compared to the manual version:

| RAG Step | Without LangChain (manual) | With LangChain (this repo) |
|---|---|---|
| PDF loading | `PyMuPDF (fitz)` — custom `load_pdf.py` | `PdfReader` from `pypdf` |
| Text splitting | Custom `chunkers.py` with overlap logic | `RecursiveCharacterTextSplitter` |
| Embedding | Direct `openai.embeddings.create()` call | `OpenAIEmbeddings` wrapper |
| Vector storage | Manual `pinecone.upsert()` with metadata | `PineconeVectorStore.from_texts()` |
| Similarity search | Manual cosine query + metadata retrieval | `vectorstore.similarity_search(k=4)` |
| LLM call | Direct `openai.chat.completions.create()` | `ChatOpenAI.invoke(messages)` |
| Prompt building | Manual f-string construction | Manual (same — LangChain prompt templates not used here) |

**Key insight:** LangChain doesn't change what RAG does — it standardises how you write it. The same 6 steps happen in both versions. LangChain just makes each step a one-liner with consistent interfaces.

---

## 📂 Project Structure

```
HR-Policy-RAG-Chatbot-using-Langchain/
│
├── llm_langchain.py     # Core RAG logic: load → chunk → embed → store → query → answer
├── app.py               # Streamlit UI — calls query_processor(), handles session state
│
├── resources/           # (gitignored) — add your HR Policy PDF here
├── .env                 # (gitignored) — API keys
├── .gitignore
├── requirements.txt
└── README.md
```

**Note on design:** Compared to the from-scratch version (8 files), this version consolidates everything into `llm_langchain.py`. This is intentional — LangChain's abstractions are clean enough that the full RAG pipeline fits in one well-structured file without becoming hard to read.

---

## 🛠️ Tech Stack

| Tool | Role | LangChain Class |
|---|---|---|
| Python | Core language | — |
| pypdf | PDF text extraction | `PdfReader` |
| LangChain | RAG orchestration framework | `langchain_*` |
| OpenAI text-embedding-3-small | 1536-dim text vectorization | `OpenAIEmbeddings` |
| OpenAI gpt-3.5-turbo | Answer generation | `ChatOpenAI` |
| Pinecone | Cloud vector database | `PineconeVectorStore` |
| Streamlit | Web UI | — |
| python-dotenv | Secure API key loading | `load_dotenv()` |

---

## ⚙️ How It Works

### Step 1 — PDF Loading
`PdfReader` from `pypdf` reads all pages and joins them into a single text string. Simple and reliable for standard HR PDF documents.

### Step 2 — Chunking with `RecursiveCharacterTextSplitter`
Splits text using `chunk_size=900` and `chunk_overlap=150`. The `Recursive` splitter tries to split on paragraphs first, then sentences, then words — preserving semantic units as much as possible. The 150-character overlap ensures context at chunk boundaries isn't lost during retrieval.

### Step 3 — Embedding + Pinecone Storage
`OpenAIEmbeddings` converts each chunk into a 1536-dimensional vector using `text-embedding-3-small`. `PineconeVectorStore.from_texts()` upserts all vectors with the original chunk text stored as metadata — one call handles both embedding and storage.

### Step 4 — Query Processing
The user's question is embedded using the same model and matched against stored vectors via cosine similarity. `similarity_search(k=4)` returns the 4 most semantically relevant chunks as LangChain `Document` objects.

### Step 5 — Answer Generation
The 4 retrieved chunks are joined and injected into a GPT-3.5-turbo prompt with a system instruction to answer only from the provided context. `temperature=0.4` keeps answers factual and consistent rather than creative.

### Step 6 — Streamlit UI
`app.py` handles user input, calls `query_processor()`, and displays the answer. Session state preserves conversation history across turns.

---

## 💬 Sample Questions & Answers

These are example queries the chatbot handles when loaded with an HR Policy document:

---

**Q: What is the leave policy for casual leave?**
> A: Employees are entitled to 12 days of casual leave per calendar year. Casual leave cannot be carried forward to the next year and must be applied at least 1 day in advance except in emergencies.

---

**Q: How many days of maternity leave are employees entitled to?**
> A: Female employees are entitled to 26 weeks of paid maternity leave as per the Maternity Benefit Act. This is applicable after completing 80 days of service in the 12 months preceding the expected delivery date.

---

**Q: What is the notice period for resignation?**
> A: The notice period is 30 days for employees below manager level and 60 days for manager level and above. The company may waive the notice period at its discretion.

---

**Q: Is work from home allowed?**
> A: Work from home is permitted up to 2 days per week for eligible roles, subject to manager approval. Employees must be reachable during core hours (10 AM – 4 PM) while working remotely.

---

**Q: What expenses are covered under the travel reimbursement policy?**
> A: Business travel expenses including airfare (economy class), hotel accommodation up to ₹4,000/night, and daily allowance of ₹500 for meals are reimbursable with supporting receipts submitted within 7 days of travel.

---

> **Note:** The above answers are illustrative. Actual answers depend on the HR Policy PDF you load. The chatbot answers strictly from the document — it will not fabricate information outside the uploaded policy.

---

## 🧠 Key Learnings — Using LangChain for RAG

### 1. `RecursiveCharacterTextSplitter` is smarter than fixed splitting
It tries paragraph breaks first, then sentence breaks, then word breaks — only splitting at character level as a last resort. This means chunks are almost always semantically complete units. I understood *why* this matters because I first saw what happens with naive splitting in the manual version.

### 2. `chunk_overlap=150` is not just a parameter — it's a retrieval quality decision
With 900-character chunks and 150-character overlap, a topic that spans two chunks will appear in at least one chunk fully. Too little overlap = missed answers at boundaries. Too much = duplicate context sent to the LLM and wasted tokens.

### 3. `temperature=0.4` for factual Q&A, not `0`
`temperature=0` makes GPT very literal — sometimes refusing to paraphrase or combine information from multiple chunks. `0.4` gives just enough flexibility to synthesise across retrieved context while staying grounded.

### 4. `PineconeVectorStore.from_texts()` handles the embedding + upsert in one call
In the manual version, these were two separate steps (embed then upsert). LangChain batches and handles both. This is the kind of abstraction that speeds up production development — but only makes sense when you already know what it's doing underneath.

### 5. LangChain's `Document` objects carry metadata automatically
`similarity_search()` returns `Document` objects with `.page_content` and `.metadata`. In the manual version, I had to manually retrieve metadata from Pinecone alongside vectors. LangChain handles this transparently — a real time-saver in production.

### 6. Fewer files ≠ worse architecture
The from-scratch version has 8 files for clear separation of concerns. This version has 2. Both are correct — LangChain's abstractions are clean enough that consolidation doesn't hurt readability. Choosing the right level of modularity for the complexity level is itself a design skill.

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/connectnataraj-boop/HR-Policy-RAG-Chatbot-using-Langchain.git
cd HR-Policy-RAG-Chatbot-using-Langchain
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Create a `.env` File
```
OPENAI_API_KEY=your_openai_key_here
PINECONE_API_KEY=your_pinecone_key_here
PINECONE_INDEX_NAME=your_index_name_here
```

### 4. Add Your HR Policy PDF
```
Create a resources/ folder in the project root
Place your HR Policy PDF inside it
Update the file path in llm_langchain.py and app.py
```

### 5. Run the Indexing Pipeline (Once)
```bash
python llm_langchain.py
```
This loads the PDF, chunks it, embeds it, and upserts to Pinecone. Only needs to run once unless the document changes.

### 6. Launch the App
```bash
streamlit run app.py
```

---


## 👤 Author

**S. Nataraj** — AI Engineer | Deep Learning & Gen AI
Tirupur, Tamil Nadu, India
📧 connectnataraj@outlook.com
🔗 [GitHub](https://github.com/connectnataraj-boop) · [LinkedIn](https://www.linkedin.com/in/nataraj-sb-b5a84a3b7/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> *"I built RAG without LangChain first. Then I built it with LangChain. Now I understand both — and that's the difference between using a framework and knowing one."*
