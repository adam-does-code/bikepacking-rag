# TODO

- [ ] Upgrade to Python >=3.10 (via Homebrew), then switch to the modern
      `google-genai` package — currently on the deprecated
      `google-generativeai` because this environment's Python 3.9 doesn't
      support the newer package.
- [ ] Fix `Handlebar Cradles` chunk ranking (currently #4/83 for a clearly
      relevant query) by prepending the heading path into the embedded text,
      not just storing it in metadata.
- [ ] Consider hybrid search (BM25 + dense embeddings via Reciprocal Rank
      Fusion) if retrieval quality becomes a bottleneck — not needed at the
      current 83-chunk scale.
- [ ] Consider a reranking model (cross-encoder second pass) — same caveat,
      only worth it at much larger corpus sizes.
