# HR Policy RAG Chatbot — Built With LangChain

> **RAG · LangChain · OpenAI · Pinecone · Cohere Rerank · MCP · Streamlit**

A Retrieval-Augmented Generation (RAG) chatbot that answers questions about an HR policy document. It is built with **LangChain**, adds a **Cohere reranking** stage, and ships with an **MCP server** that exposes the chatbot's tools to any MCP-capable AI app.

> 🚀 **Live demo:** [hr-policy-rag-chatbot-using-langchain.streamlit.app](https://hr-policy-rag-chatbot-using-langchain.streamlit.app)
>
> 🔗 Also see the from-scratch version: [HR-Policy-RAG-Chatbot-without-Langchain](https://github.com/connectnataraj-boop/HR-Policy-RAG-Chatbot-without-Langchain)

---

## 📌 Table of Contents

- [Why I Built Two Versions](#-why-i-built-two-versions)
- [What's New: Reranking, Caching, and Metrics](#-whats-new-reranking-caching-and-metrics)
- [Architecture](#-architecture)
- [Evaluation Results](#-evaluation-results)
- [MCP Server](#-mcp-server)
- [LangChain vs From Scratch](#-langchain-vs-from-scratch--what-each-abstracts)
- [Project Structure](#-project-structure)
- [Tech Stack](#-tech-stack)
- [Example Questions](#-example-questions)
- [Key Learnings](#-key-learnings)
- [Setup Instructions](#-setup-instructions)
- [Known Limitations](#-known-limitations)
- [Author](#-author)

---

## 🤔 Why I Built Two Versions

Most people learn RAG one of two ways:

- **Only LangChain** → fast to build, but you don't understand what's happening inside
- **Only from scratch** → deep understanding, but not how teams usually build it

I built **both** — in the right order:

| Version | Approach | What I Learned |
|---|---|---|
| [Without LangChain](https://github.com/connectnataraj-boop/HR-Policy-RAG-Chatbot-without-Langchain) | Every component handwritten | How RAG works under the hood |
| **This repo** (With LangChain) | LangChain abstractions | How teams typically build it, and how to measure and improve it |

Building the from-scratch version first meant I arrived here already knowing what `RecursiveCharacterTextSplitter`, `PineconeVectorStore`, and `similarity_search` are doing. That helps me debug failures, tune parameters, and reason about trade-offs instead of calling functions blindly.

---

## ✨ What's New: Reranking, Caching, and Metrics

| Feature | What it does |
|---|---|
| **Cohere reranking** | Retrieves 20 chunks from Pinecone, re-scores them against the question, and sends only the best 4 to the LLM. If the Cohere call fails or no key is set, it falls back to normal vector order. |
| **PDF upload** | Users upload their own policy PDF in the UI. Each file is indexed in its own Pinecone namespace (an MD5 hash of the file), so documents do not mix and re-uploads do not create duplicates. |
| **Answer cache** | Repeated questions are answered from an in-memory cache with no API calls. Exact match only, up to 200 entries. |
| **Token, cost, and time metrics** | Every answer shows input and output tokens, estimated cost, and the time spent in search, rerank, and the LLM. A sidebar shows session totals. |
| **Sidebar controls** | Turn reranking and caching on or off, and set how many chunks are retrieved and sent to the LLM. This makes before-and-after comparisons easy. |
| **Output cap** | Answers are limited to 300 tokens and the prompt asks for concise answers, which reduces time and cost. |
| **MCP server** | Exposes three read-only HR tools over the Model Context Protocol. See [MCP Server](#-mcp-server). |

---

## 🏗️ Architecture

### Indexing (runs when a PDF is uploaded)

```
HR Policy PDF (uploaded in the UI)
      │
      ▼
┌──────────────────────────────────────┐
│  PdfReader (pypdf)                   │  extracts raw text from all pages
│  load_pdf()                          │
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  RecursiveCharacterTextSplitter      │  chunk_size=900, chunk_overlap=150
│  get_chunks()                        │  paragraphs → sentences → words
└──────────────────┬───────────────────┘
                   ▼
┌──────────────────────────────────────┐
│  OpenAIEmbeddings                    │  text-embedding-3-small (1536 dims)
│  PineconeVectorStore.from_texts()    │  namespace = MD5 of the file
└──────────────────────────────────────┘
```

### Query pipeline (every question)

```
User question → Streamlit UI (app.py)
        │
        ▼
  Cache hit? ── yes ──► return saved answer (no API calls)
        │ no
        ▼
  similarity_search (k=20)        ← Pinecone, OpenAI embeddings
        │
        ▼
  Cohere Rerank → best 4 chunks   ← falls back to vector order on failure
        │
        ▼
  ChatOpenAI (gpt-4o-mini, temperature 0.4, max 300 tokens)
  system: "Answer only from the context, say you don't know otherwise"
        │
        ▼
  Answer + stats (tokens, cost, search / rerank / LLM time) → Streamlit UI
```

---

## 📊 Evaluation Results

I wrote a **15-question test set** from the policy document, including questions the document cannot answer (such as the number of casual leave days, which the policy does not state). I scored each answer Pass, Partial, or Fail (Pass = 1, Partial = 0.5, and an invented number counts as a Fail).

| | Before | After |
|---|---|---|
| Correct answers (own scoring) | 10 / 15 | 13.5 / 15 |
| Average response time (one run per question) | 1.97 s | 1.53 s |

**How to read this honestly**
- The four answers that went from Fail to Pass were mostly retrieval problems, where the right chunk was not in the top results.
- Between the two runs I changed **several things at once**: reranking, the prompt wording, the model (gpt-3.5-turbo to gpt-4o-mini), and the number of chunks sent. So the improvement is not from reranking alone. A fair test would change one setting at a time, and the sidebar controls are there for that.
- The time results are **one run per question** on my own network, so differences of a few tenths of a second are within normal noise.
- The test document is small (about 5 to 6 chunks), so reranking has little room to help. A larger document set would be a better test.
- The scoring is mine and manual. Tools such as RAGAS or LangSmith would be the next step.

---

## 🔌 MCP Server

`mcp_server.py` is a small [Model Context Protocol](https://modelcontextprotocol.io) server built with the Python MCP SDK. It lets any MCP-capable AI app discover and call the chatbot's HR tools.

| Tool | What it does | Uses AI? |
|---|---|---|
| `get_leave_policy(leave_type)` | Returns fixed policy facts for annual (earned), casual, sick, or compensatory leave | No |
| `search_hr_policy(query, top_n=4)` | Retrieves 20 passages from Pinecone, reranks them with Cohere, returns the best `top_n` | Embeddings + rerank |
| `ask_hr_policy(question)` | Answers a question from the HR policy only, using the full RAG pipeline | Yes |

**Test it**

```bash
pip install "mcp[cli]<2"
python test_mcp_client.py
```

The client starts the server, lists the tools, and calls each one. Startup takes 10 to 20 seconds, because the server connects to Pinecone before it accepts calls.

**Design choices**
- **Read-only tools.** Nothing can change or delete data, and any write action would need human approval.
- **Fixed lookup for known facts,** so `get_leave_policy` is instant, free, and exact.
- **Input checks** reject empty or very long inputs.
- **Fail-safe errors.** A failing tool returns a safe message and logs the cause to stderr, so the server does not crash.
- **Slow work runs in a worker thread** with a 90 second limit, so one slow call cannot freeze the server.
- **Logs go to stderr only,** because standard output carries the MCP messages.

---

## 🔄 LangChain vs From Scratch — What Each Abstracts

| RAG Step | Without LangChain (manual) | With LangChain (this repo) |
|---|---|---|
| PDF loading | Custom `load_pdf.py` using `pypdf` | `PdfReader` from `pypdf` |
| Text splitting | Custom `chunkers.py` with overlap logic | `RecursiveCharacterTextSplitter` |
| Embedding | Direct `openai.embeddings.create()` call | `OpenAIEmbeddings` wrapper |
| Vector storage | Manual `index.upsert()` with metadata | `PineconeVectorStore.from_texts()` |
| Similarity search | Manual cosine query + metadata retrieval | `vectorstore.similarity_search(k=...)` |
| LLM call | Direct `openai.chat.completions.create()` | `ChatOpenAI.invoke(messages)` |
| Prompt building | Manual f-string construction | Manual (LangChain prompt templates not used here) |

**Key insight:** LangChain does not change what RAG does. It standardises how you write it. The same steps happen in both versions.

---

## 📂 Project Structure

```
HR-Policy-RAG-Chatbot-using-Langchain/
│
├── llm_langchain.py     # RAG logic: load → chunk → embed → store → retrieve → rerank → answer, plus cache and metrics
├── app.py               # Streamlit UI: PDF upload, sidebar controls, answers with stats
├── mcp_server.py        # MCP server exposing three read-only HR tools
├── test_mcp_client.py   # Python MCP client that starts the server and calls each tool
│
├── resources/           # (gitignored) HR policy PDF used by the MCP server
├── .env                 # (gitignored) API keys
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🛠️ Tech Stack

| Tool | Role | LangChain Class |
|---|---|---|
| Python | Core language | — |
| pypdf | PDF text extraction | `PdfReader` |
| LangChain | RAG orchestration | `langchain_*` |
| OpenAI `text-embedding-3-small` | 1536-dim text vectors | `OpenAIEmbeddings` |
| OpenAI `gpt-4o-mini` | Answer generation | `ChatOpenAI` |
| Pinecone | Cloud vector database | `PineconeVectorStore` |
| Cohere Rerank | Re-orders retrieved chunks by relevance | — (called directly) |
| MCP Python SDK | MCP server and test client | — |
| Streamlit | Web UI | — |
| python-dotenv | API key loading | `load_dotenv()` |

---

## 💬 Example Questions

These are real questions from my test set, answered from the policy document I used:

| Question | Answer from the document |
|---|---|
| How many days of earned leave do I get per year? | 24 days per year, credited monthly |
| How many earned leave days can be carried forward? | Up to 15 days; the rest lapse |
| Can earned leave be encashed while employed? | No, only on separation from the company |
| How long do I have to use a compensatory off? | Within thirty days of accrual |
| How many casual leave days are there? | The policy does not state a number, and the chatbot says so |

The chatbot answers only from the uploaded document. When the document does not contain the answer, it says it does not know.

---

## 🧠 Key Learnings

1. **`RecursiveCharacterTextSplitter` is smarter than fixed splitting.** It tries paragraph breaks first, then sentences, then words, so chunks are mostly complete ideas.
2. **`chunk_overlap=150` is a retrieval quality decision.** Too little overlap misses answers at chunk boundaries. Too much wastes tokens on duplicate context.
3. **Most wrong answers are retrieval problems.** Checking which chunks came back is the first debugging step, before changing the prompt.
4. **Reranking is a second stage, not a replacement for retrieval.** Retrieve wide, rerank, and keep a few. It can add some latency, so measure it.
5. **Change one thing at a time when you evaluate.** I changed the model, prompt, and retrieval together in my first comparison, which made the result harder to explain.
6. **Measure cost and time, not just accuracy.** Showing tokens and time per stage showed where the time actually goes.
7. **MCP separates the tool from the app.** Wrapping the chatbot's functions once as an MCP server means any MCP-capable app can reuse them, and it stays read-only and safe by design.

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
COHERE_API_KEY=your_cohere_key_here

# Optional: for cost estimates, USD per 1 million tokens (check OpenAI's pricing page)
PRICE_IN_PER_1M=0
PRICE_OUT_PER_1M=0
```

The Pinecone index must use dimension **1536** and the cosine metric (for `text-embedding-3-small`). Cohere is optional: without a key, the app skips reranking and uses vector order.

### 4. Launch the App
```bash
streamlit run app.py
```
Upload your HR policy PDF in the UI. The first upload is chunked, embedded, and stored in Pinecone automatically.

### 5. (Optional) Run the MCP server test
Put your policy PDF at `resources/HRPolicy.pdf`, then run `python test_mcp_client.py`.

### Deploying on Streamlit Cloud
Add the same keys under **Settings → Secrets** and list `cohere` and `langchain-pinecone` in `requirements.txt`.

---

## ⚠️ Known Limitations

- Vector search plus reranking only. There is no keyword (hybrid) search.
- The cache is in memory and matches exact questions only. It resets when the app restarts.
- The MCP server runs locally over stdio. It has no authentication and is not deployed.
- PDF text extraction keeps extra line breaks, so retrieved passages look messy in the MCP output. Answers are still correct, and cleaning the whitespace before chunking is the next fix.
- The evaluation is a small, manual 15-question test on one short document.
- Only PDF files are supported.

---

## 👤 Author

**S. Nataraj** — GenAI Engineer | LangGraph, RAG, Agents
Tirupur, Tamil Nadu, India
📧 connectnataraj@outlook.com
🔗 [GitHub](https://github.com/connectnataraj-boop) · [LinkedIn](https://www.linkedin.com/in/nataraj-sb-b5a84a3b7/)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

> *"I built RAG without LangChain first. Then I built it with LangChain. Then I measured it and added reranking. That's the difference between using a framework and understanding one."*
