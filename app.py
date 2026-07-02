"""
app.py

Context-Aware Customer Support RAG Bot.

Terminal-based chatbot loop that:
  1. Accepts a user_id and user_query.
  2. Looks up the user's name and membership tier from users.db.
  3. Retrieves the most relevant FAQ chunks from the local FAISS vector store.
  4. Builds a grounded prompt and calls the Groq API (llama3-8b-8192) to
     generate a personalized, hallucination-safe answer.

Run:
    python create_db.py     # once, to create/seed users.db
    python ingest.py        # once, to build the vector store
    python app.py           # start the chatbot loop
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
        row = cursor.fetchone()
        return row  # (name, membership_tier) or None
    finally:
        connection.close()


def load_vector_store():
    """Load the persisted FAISS vector store from disk."""
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
    return vector_store


def retrieve_context(vector_store, user_query: str, k: int = TOP_K_CHUNKS):
    """Return the top-k relevant chunks (with a relevance-score cutoff)."""
    results = vector_store.similarity_search_with_score(user_query, k=k)

    # FAISS uses L2 distance here: lower = more similar. Filter out
    # weak/irrelevant matches so unrelated questions correctly fall back
    # to the "not enough information" response instead of hallucinating.
    SCORE_THRESHOLD = 1.0
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
            "GROQ_API_KEY is not set. Copy .env.example to .env and add your "
            "Groq API key, e.g. GROQ_API_KEY=your_actual_key_here."
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


def run_cli():
    print("=" * 60)
    print("NimbusCart Customer Support Assistant")
    print("=" * 60)
    print("Type 'exit' at any prompt to quit.\n")

    try:
        vector_store = load_vector_store()
        client = get_groq_client()
    except ConfigurationError as exc:
        print(f"Setup error: {exc}")
        return

    while True:
        raw_user_id = input("Enter user_id: ").strip()
        if raw_user_id.lower() == "exit":
            break
        try:
            user_id = int(raw_user_id)
        except ValueError:
            print(USER_NOT_FOUND_MESSAGE)
            print()
            continue

        user_query = input("Enter your question: ").strip()
        if user_query.lower() == "exit":
            break
        if not user_query:
            print("Please enter a non-empty question.\n")
            continue

        answer = answer_query(vector_store, client, user_id, user_query)
        print(f"\nAssistant: {answer}\n")
        print("-" * 60)


if __name__ == "__main__":
    run_cli()
