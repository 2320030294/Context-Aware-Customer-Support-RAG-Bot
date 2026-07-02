"""
create_db.py

Creates the local SQLite database (users.db) used by the RAG chatbot to
personalize answers based on the requesting user's membership tier.

Run this once before starting app.py:
    python create_db.py
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent / "users.db"

SAMPLE_USERS = [
    (101, "Riya Sharma", "Gold"),
    (102, "Aman Verma", "Silver"),
    (103, "Neha Iyer", "Platinum"),
]


def create_and_seed_database(db_path: Path = DB_PATH) -> None:
    """Create the users table (if needed) and seed it with sample rows."""
    connection = sqlite3.connect(db_path)
    try:
        cursor = connection.cursor()

        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                user_id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                membership_tier TEXT NOT NULL
            )
            """
        )

        # Use INSERT OR REPLACE so re-running this script is safe/idempotent.
        cursor.executemany(
            """
            INSERT OR REPLACE INTO users (user_id, name, membership_tier)
            VALUES (?, ?, ?)
            """,
            SAMPLE_USERS,
        )

        connection.commit()
        print(f"Database ready at: {db_path}")
        print(f"Seeded {len(SAMPLE_USERS)} users:")
        for user_id, name, tier in SAMPLE_USERS:
            print(f"  - {user_id}: {name} ({tier})")
    finally:
        connection.close()


if __name__ == "__main__":
    create_and_seed_database()
