"""
rag_pipeline.py

Shared RAG pipeline logic used by both app.py (terminal CLI) and
streamlit_app.py (hosted web UI), so retrieval, prompting, and error
handling behave identically regardless of which interface is used.
"""

import os
import sqlite3
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent
DB_PATH = BASE_DIR / "users.db"
VECTOR_STORE_DIR = BASE_DIR / "faiss_index"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL_NAME = "llama3-8b-8192"
TOP_K_CHUNKS = 3
SCORE_THRESHOLD = 1.0  # FAISS L2 distance; lower = more similar

NO_CONTEXT_MESSAGE = (
    "I do not have enough information in the provided knowledge base to answer this."
)
USER_NOT_FOUND_MESSAGE = "User not found. Please enter a valid user_id."

PROMPT_TEMPLATE = """You are an AI customer support assistant.

You are speaking with:
Name: {name}
Membership Tier: {membership_tier}

Answer the user's question using only the context provided below.

If the answer is not available in the context, say:
"I do not have enough information in the provided knowledge base to answer this."

Context:
{retrieved_chunks}

User Question:
{user_query}

Answer:"""


class ConfigurationError(Exception):
    """Raised when required setup (API key, vector store, etc.) is missing."""


def ensure_setup():
    """
    Auto-bootstrap: build users.db and the FAISS index on first run if
    they don't exist yet. This lets the app deploy cleanly on a fresh
    clone/hosting platform without needing users.db or faiss_index/ to be
    committed to git — they're generated automatically the first time the
    app starts, then reused on every subsequent run.
    """
    if not DB_PATH.exists():
        from create_db import create_and_seed_database
        create_and_seed_database()

    if not VECTOR_STORE_DIR.exists():
        from ingest import main as run_ingest
        run_ingest()


def get_user(user_id: int):
    """Fetch (name, membership_tier) for a user_id, or None if not found."""
    if not DB_PATH.exists():
        raise ConfigurationError(
            f"users.db not found at {DB_PATH}. Run 'python create_db.py' first."
        )

    connection = sqlite3.connect(DB_PATH)
    try:
        cursor = connection.cursor()
        cursor.execute(
            "SELECT name, membership_tier FROM users WHERE user_id = ?",
            (user_id,),
        )
        return cursor.fetchone()  # (name, membership_tier) or None
    finally:
        connection.close()


_vector_store_cache = None


def load_vector_store(use_cache: bool = True):
    """Load the persisted FAISS vector store from disk (cached in-process)."""
    global _vector_store_cache
    if use_cache and _vector_store_cache is not None:
        return _vector_store_cache

    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    if not VECTOR_STORE_DIR.exists():
        raise ConfigurationError(
            f"Vector store not found at {VECTOR_STORE_DIR}. "
            "Run 'python ingest.py' first."
        )

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_DIR),
        embeddings,
        allow_dangerous_deserialization=True,
    )
    if use_cache:
        _vector_store_cache = vector_store
    return vector_store


def retrieve_context(vector_store, user_query: str, k: int = TOP_K_CHUNKS):
    """Return the top-k relevant chunks (with a relevance-score cutoff)."""
    results = vector_store.similarity_search_with_score(user_query, k=k)
    relevant_docs = [doc for doc, score in results if score <= SCORE_THRESHOLD]

    if not relevant_docs:
        return None

    return "\n\n".join(doc.page_content for doc in relevant_docs)


def get_groq_client():
    """Instantiate the Groq client, raising a clear error if misconfigured."""
    from groq import Groq

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ConfigurationError(
            "GROQ_API_KEY is not set. Add it to your .env file (local) or to "
            "your hosting platform's secrets/environment variables (deployed)."
        )
    return Groq(api_key=api_key)


def generate_answer(client, name: str, membership_tier: str,
                     retrieved_chunks: str, user_query: str) -> str:
    """Call the Groq API to generate a grounded, personalized answer."""
    prompt = PROMPT_TEMPLATE.format(
        name=name,
        membership_tier=membership_tier,
        retrieved_chunks=retrieved_chunks,
        user_query=user_query,
    )

    try:
        response = client.chat.completions.create(
            model=GROQ_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:  # Broad catch: surface a readable message, never crash.
        error_text = str(exc).lower()
        if "rate limit" in error_text or "429" in error_text:
            return ("The support assistant is currently rate-limited. "
                     "Please try again in a moment.")
        if "authentication" in error_text or "401" in error_text or "api key" in error_text:
            return ("There was an authentication error with the Groq API. "
                     "Please check that GROQ_API_KEY is set correctly.")
        return f"Sorry, something went wrong while generating a response: {exc}"


def answer_query(vector_store, client, user_id: int, user_query: str) -> str:
    """Full pipeline for a single (user_id, user_query) request."""
    user = get_user(user_id)
    if user is None:
        return USER_NOT_FOUND_MESSAGE

    name, membership_tier = user

    retrieved_chunks = retrieve_context(vector_store, user_query)
    if not retrieved_chunks:
        return NO_CONTEXT_MESSAGE

    return generate_answer(client, name, membership_tier, retrieved_chunks, user_query)
