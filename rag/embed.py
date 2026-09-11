"""
Step 3 of the RAG pipeline: embedding.

Loads the chunks produced by chunk.py and converts each chunk's text into
a dense vector using a local sentence-transformers model — no API key,
no network calls after the model weights are downloaded once.
"""

import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

CHUNKS_PATH = Path(__file__).parent / "chunks.json"
OUT_PATH = Path(__file__).parent / "embeddings.json"
MODEL_NAME = "all-MiniLM-L6-v2"


def main():
    chunks = json.loads(CHUNKS_PATH.read_text())
    print(f"Loaded {len(chunks)} chunks")

    print(f"Loading model '{MODEL_NAME}' (downloads once, then cached locally)...")
    model = SentenceTransformer(MODEL_NAME)
    dim = model.get_sentence_embedding_dimension()
    print(f"Embedding dimension: {dim}")

    texts = [c["text"] for c in chunks]
    # normalize_embeddings=True makes cosine similarity == dot product,
    # which simplifies the similarity search we'll do in the next step.
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)

    for chunk, vector in zip(chunks, embeddings):
        chunk["embedding"] = vector.tolist()

    OUT_PATH.write_text(json.dumps(chunks))
    print(f"\nWrote {len(chunks)} embedded chunks to {OUT_PATH}")


if __name__ == "__main__":
    main()
