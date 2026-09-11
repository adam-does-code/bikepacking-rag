# bikepacking-rag

A small retrieval-augmented generation (RAG) project built as a learning exercise,
answering questions over the [BIKEPACKING.com Bikepacking 101 Handbook](https://bikepacking.com/bikepacking-101/).

## Pipeline

1. **Parsing** — the 5 handbook pages were fetched and hand-cleaned into markdown
   (stripping nav, ads, and promotional content). Not included in this repo — see
   Content note below.
2. **Chunking** ([`rag/chunk.py`](rag/chunk.py)) — splits each markdown file into
   chunks along its `##`/`###` headings rather than fixed token windows, so each
   chunk stays semantically whole and carries a clean heading path (e.g.
   `Bikepacking Bags and How to Pack > Handlebar Bags > Handlebar Cradles`) usable
   as a citation. Oversized sections are further split on paragraph breaks.
3. **Embedding** ([`rag/embed.py`](rag/embed.py)) — converts each chunk's text into
   a 384-dimension vector with a local `sentence-transformers` model
   (`all-MiniLM-L6-v2`) — no API key, no network calls after the model weights are
   cached. Vectors are L2-normalized so cosine similarity reduces to a dot product.
4. **Vector storage** — TODO
5. **Query + retrieval** — TODO

## Content note

The scraped handbook text (`bikepacking-101/*.md`, `rag/chunks.json`,
`rag/embeddings.json`) is intentionally excluded from this repo (see
`.gitignore`). BIKEPACKING.com's [Terms of Use](https://bikepacking.com/about/privacy/)
prohibit crawling, harvesting, or scraping site content, so this repo publishes
only the original pipeline code, not the underlying content it was built and
tested against.

## Setup

Requires Python >=3.9,<3.10 with this project's pinned dependency versions —
`torch<2.3` and `sentence-transformers<3.0` are capped for compatibility with
older Python/macOS combinations; `numpy<2` is capped to match the ABI `torch`
was compiled against. Adjust these caps if your environment supports newer
versions.

```bash
poetry install
poetry run python rag/chunk.py
poetry run python rag/embed.py
```
