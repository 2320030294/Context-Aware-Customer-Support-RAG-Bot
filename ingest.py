"""
ingest.py

Reads the company FAQ document, splits it into overlapping chunks,
generates embeddings with a local/free HuggingFace embedding model, and
persists a FAISS vector store to disk so app.py can load it quickly at
query time without re-embedding on every run.

Run this once (and again any time company_faq.txt changes):
    python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from rag_pipeline import EMBEDDING_MODEL_NAME, VECTOR_STORE_DIR

BASE_DIR = Path(__file__).parent
FAQ_PATH = BASE_DIR / "company_faq.txt"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def load_and_chunk_faq(faq_path: Path = FAQ_PATH):
    """Load the FAQ text file and split it into overlapping chunks."""
    if not faq_path.exists():
        raise FileNotFoundError(
            f"Could not find FAQ document at {faq_path}. "
            "Make sure company_faq.txt exists before running ingest.py."
        )

    loader = TextLoader(str(faq_path), encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    return chunks


def build_vector_store(chunks, persist_dir: Path = VECTOR_STORE_DIR):
    """Embed chunks and persist a FAISS index to disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(persist_dir))
    return vector_store


def main():
    print(f"Loading and chunking: {FAQ_PATH}")
    chunks = load_and_chunk_faq()
    print(f"Created {len(chunks)} chunks.")

    print(f"Generating embeddings with '{EMBEDDING_MODEL_NAME}' "
          "(first run downloads the model, this may take a minute)...")
    build_vector_store(chunks)

    print(f"Vector store saved to: {VECTOR_STORE_DIR}")
    print("Ingestion complete. You can now run: python app.py")


if __name__ == "__main__":
    main()
