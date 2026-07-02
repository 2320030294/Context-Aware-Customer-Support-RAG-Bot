"""
app.py

Context-Aware Customer Support RAG Bot — terminal-based interface.

Run:
    python create_db.py     # once, to create/seed users.db
    python ingest.py        # once, to build the vector store
    python app.py           # start the chatbot loop
"""

from rag_pipeline import (
    ConfigurationError,
    answer_query,
    ensure_setup,
    get_groq_client,
    load_vector_store,
    USER_NOT_FOUND_MESSAGE,
)


def run_cli():
    print("=" * 60)
    print("NimbusCart Customer Support Assistant")
    print("=" * 60)
    print("Type 'exit' at any prompt to quit.\n")

    try:
        ensure_setup()
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
