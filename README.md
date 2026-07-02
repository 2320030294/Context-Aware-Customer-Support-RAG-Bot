# Context-Aware Customer Support RAG Bot

A Retrieval-Augmented Generation (RAG) chatbot that answers customer support
questions from a company FAQ document and personalizes each answer using
user details (name, membership tier) stored in a local SQLite database.

**Stack:** Python · LangChain · Groq API (`llama3-8b-8192`) · SQLite ·
FAISS · HuggingFace embeddings (`sentence-transformers/all-MiniLM-L6-v2`)

---

## Project Structure

```
.
├── app.py              # Main chatbot application (terminal loop)
├── ingest.py            # Chunks company_faq.txt and builds the FAISS vector store
├── create_db.py          # Creates and seeds users.db (SQLite)
├── company_faq.txt        # Sample FAQ knowledge base
├── requirements.txt
├── .env.example
├── README.md
├── users.db              # Created after running create_db.py
└── faiss_index/           # Created after running ingest.py
```

## 1. Setup

### Create a virtual environment (recommended)

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

### Configure your Groq API key

```bash
cp .env.example .env
```

Then edit `.env` and replace the placeholder with your real key:

```
GROQ_API_KEY=your_actual_groq_api_key
```

Get a free key from [console.groq.com](https://console.groq.com/).

## 2. Build the database and vector store

Run these two one-time setup scripts, in order:

```bash
python create_db.py
python ingest.py
```

- `create_db.py` creates `users.db` and seeds it with 3 sample users
  (Riya Sharma / Gold, Aman Verma / Silver, Neha Iyer / Platinum).
- `ingest.py` reads `company_faq.txt`, splits it into overlapping chunks,
  embeds them locally with a free HuggingFace model, and saves a FAISS
  index to `faiss_index/`. The first run downloads the embedding model
  (~90MB) and may take a minute; it's cached locally after that.

## 3. Run the chatbot

```bash
python app.py
```

You'll be prompted for a `user_id` and a question, in a loop. Type `exit`
at either prompt to quit.

## Sample Queries to Try

| user_id | Question | Expected behavior |
|---|---|---|
| 101 | `What is the refund policy?` | Answers using the refund policy section, referencing Riya Sharma's Gold-tier 45-day return window. |
| 103 | `Do I get premium customer support?` | Answers using the premium support rules, referencing Neha Iyer's Platinum-tier 24/7 support and relationship manager. |
| 999 | `What are my benefits?` | `User not found. Please enter a valid user_id.` |
| 102 | `Can I cancel my account?` | Answers using only the account cancellation section for Aman Verma. |
| 101 | `What's the weather today?` | `I do not have enough information in the provided knowledge base to answer this.` (out-of-scope query, no relevant chunks retrieved) |

## How It Works

1. **Ingestion (`ingest.py`)** — loads `company_faq.txt`, splits it into
   ~500-character overlapping chunks with `RecursiveCharacterTextSplitter`,
   embeds each chunk with a local HuggingFace sentence-transformer model,
   and persists the resulting FAISS index to disk.
2. **Query time (`app.py`)**:
   - Looks up `user_id` in `users.db` to get the user's `name` and
     `membership_tier`.
   - Embeds the `user_query` and retrieves the top-3 most similar chunks
     from the FAISS index, filtering out low-relevance matches so
     off-topic questions correctly trigger the "not enough information"
     fallback instead of a hallucinated answer.
   - Builds a strict, context-only prompt (see `PROMPT_TEMPLATE` in
     `app.py`) and sends it to the Groq API using `llama3-8b-8192`.
   - Returns the model's answer, personalized to the user's name and tier.

## Error Handling

| Scenario | Behavior |
|---|---|
| `user_id` not in database | `User not found. Please enter a valid user_id.` |
| No relevant chunks retrieved | `I do not have enough information in the provided knowledge base to answer this.` |
| `GROQ_API_KEY` missing/not set | Clear setup error at startup, asking you to configure `.env`. |
| Groq API error / rate limit | Caught and shown as a readable message; the app does not crash. |
| `users.db` or `faiss_index/` missing | Clear message telling you to run `create_db.py` / `ingest.py` first. |

## Notes

- The retrieval relevance cutoff (`SCORE_THRESHOLD` in `app.py`) uses FAISS
  L2 distance. If your embedding model or chunking changes, you may need
  to retune this threshold — lower distance = more similar.
- Swap FAISS for Chroma by changing the vector store import in `ingest.py`
  and `app.py` if preferred; the rest of the pipeline is unaffected.
- `app.py` is a terminal loop by design per the assessment spec ("this can
  be either a terminal-based chatbot loop, or a simple FastAPI endpoint").
  Wrapping `answer_query()` in a FastAPI route would be a small, isolated
  change if an HTTP API is needed later.
