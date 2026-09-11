"""
Step 5 of the RAG pipeline: query + retrieval.

Takes a question, embeds it, retrieves the most similar chunks from the
Chroma database, and asks Gemini (free tier) to answer using only that
retrieved context. Sources are built deterministically from chunk metadata
rather than trusted to the model, so citations can't be hallucinated.
"""

import sys

import pysqlite3

sys.modules["sqlite3"] = pysqlite3

import os
from pathlib import Path

import chromadb
import google.generativeai as genai
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

load_dotenv()

DB_DIR = str(Path(__file__).parent / "chroma_db")
COLLECTION_NAME = "bikepacking_101"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
GEMINI_MODEL = "gemini-3.6-flash"
TOP_K = 5

SYSTEM_PROMPT = """You answer questions about bikepacking using only the \
context provided below, drawn from the BIKEPACKING.com Bikepacking 101 \
handbook. If the context doesn't contain enough information to answer, say \
so plainly instead of guessing or using outside knowledge."""


def retrieve(question: str, embedder: SentenceTransformer, collection):
    query_vec = embedder.encode([question], normalize_embeddings=True).tolist()
    results = collection.query(query_embeddings=query_vec, n_results=TOP_K)
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        chunks.append({"text": doc, "metadata": meta, "distance": dist})
    return chunks


def build_context(chunks):
    blocks = []
    for i, c in enumerate(chunks, 1):
        blocks.append(f"[{i}] ({c['metadata']['section']})\n{c['text']}")
    return "\n\n".join(blocks)


def answer(question: str, chunks, model: genai.GenerativeModel) -> str:
    context = build_context(chunks)
    user_message = f"Context:\n{context}\n\nQuestion: {question}"

    response = model.generate_content(user_message)
    return response.text


def format_sources(chunks) -> str:
    seen = set()
    lines = []
    for c in chunks:
        key = c["metadata"]["source_url"]
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- {c['metadata']['section']} ({key})")
    return "\n".join(lines)


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print(
            "GEMINI_API_KEY is not set. Create a .env file in the project "
            "root with:\n  GEMINI_API_KEY=your-key-here\nor export it in "
            "your shell before running this script."
        )
        sys.exit(1)

    question = " ".join(sys.argv[1:]) or input("Ask a bikepacking question: ")

    print("Embedding question and searching...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    chroma_client = chromadb.PersistentClient(path=DB_DIR)
    collection = chroma_client.get_collection(COLLECTION_NAME)
    chunks = retrieve(question, embedder, collection)

    print("Asking Gemini...")
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(GEMINI_MODEL, system_instruction=SYSTEM_PROMPT)
    result = answer(question, chunks, model)

    print(f"\n{result}\n")
    print("Sources:")
    print(format_sources(chunks))


if __name__ == "__main__":
    main()
