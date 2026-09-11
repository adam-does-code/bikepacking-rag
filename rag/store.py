"""
Step 4 of the RAG pipeline: vector storage.

Loads the embedded chunks from embed.py and indexes them in a local Chroma
vector database, so similarity search is a library call instead of a
hand-written loop over a JSON file.

Note: this machine's system sqlite3 is too old for Chroma (needs >=3.35.0),
so we swap in pysqlite3 (built against a newer libsqlite3 via Homebrew)
before chromadb is imported. See README for the environment-specific setup.
"""

import sys

import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import json
from pathlib import Path

import chromadb

EMBEDDINGS_PATH = Path(__file__).parent / "embeddings.json"
DB_DIR = Path(__file__).parent / "chroma_db"
COLLECTION_NAME = "bikepacking_101"


def main():
    chunks = json.loads(EMBEDDINGS_PATH.read_text())
    print(f"Loaded {len(chunks)} embedded chunks")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    # Delete any existing collection so re-running this script is idempotent
    # rather than appending duplicates on top of a previous run.
    client.delete_collection(COLLECTION_NAME) if COLLECTION_NAME in [
        c.name for c in client.list_collections()
    ] else None
    collection = client.create_collection(COLLECTION_NAME)

    collection.add(
        ids=[c["id"] for c in chunks],
        embeddings=[c["embedding"] for c in chunks],
        documents=[c["text"] for c in chunks],
        metadatas=[c["metadata"] for c in chunks],
    )

    print(f"Indexed {collection.count()} chunks into '{COLLECTION_NAME}' at {DB_DIR}")


if __name__ == "__main__":
    main()
