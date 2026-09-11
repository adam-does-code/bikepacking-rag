# Chunking

Chunking splits a document into smaller, independently-retrievable pieces
before embedding — small enough to embed and return as focused context, but
large enough to still make sense on their own when retrieved out of context.

| | Ours (`rag/chunk.py`) | LangChain | NLTK | Semantic chunking |
|---|---|---|---|---|
| Splits on | Markdown headers, then paragraphs | Markdown headers, then chars (configurable) | Sentence boundaries (Punkt model) | Embedding similarity between sentences |
| Overlap between chunks | None | Configurable, on by default | Optional, sentence-level | None — boundaries are similarity drops |
| Structure-aware | Yes | Yes | No — purely linguistic | No — meaning-aware instead |
| Dependency | None (stdlib) | `langchain-text-splitters` | `nltk` + Punkt model | An embedding model (run at chunk time, not just embed time) |
| Best for | Docs with reliable headers (our case) | Same, at production scale | Unstructured prose with no headers | Unstructured prose where topic shifts matter more than sentence/word counts |
| Note | Hand-rolled version of LangChain's pattern | `MarkdownHeaderTextSplitter` + `RecursiveCharacterTextSplitter` | `nltk.chunk` is actually grammatical parsing (NP/VP), a different task — sentence tokenization is the relevant piece | Most compute-expensive — embeds during chunking, then again at the embed step |
