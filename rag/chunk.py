"""
Step 2 of the RAG pipeline: chunking.

Splits each already-parsed markdown file into section-sized chunks using
its ## / ### headers as natural boundaries, instead of blindly cutting
every N tokens. Each chunk carries metadata (source file, source URL,
heading path) so later steps can cite exactly where it came from.

If a single section is still too long, it's further split on paragraph
breaks (never mid-sentence) so no chunk grows unboundedly.
"""

import json
import re
from pathlib import Path

SRC_DIR = Path(__file__).parent.parent / "bikepacking-101"
OUT_PATH = Path(__file__).parent / "chunks.json"

MAX_WORDS = 220  # soft cap before a section gets split further

SOURCE_URLS = {
    "01-what-is-bikepacking.md": "https://bikepacking.com/bikepacking-101/",
    "02-setups.md": "https://bikepacking.com/bikepacking-101/setups",
    "03-bags.md": "https://bikepacking.com/bikepacking-101/bags",
    "04-what-to-pack.md": "https://bikepacking.com/bikepacking-101/pack-list",
    "05-route-planning.md": "https://bikepacking.com/bikepacking-101/route-planning",
}

HEADER_RE = re.compile(r"^(#{1,3})\s+(.*)")


def split_into_sections(text: str):
    """Walk the markdown line by line, using headers to build a heading
    path (e.g. ["Bikepacking Setups", "Frame Packs", "Half Frame Bag"])
    and yielding (path, body) for each leaf section."""
    sections = []
    stack = []          # list of (level, title)
    body_lines = []

    def flush():
        body = "\n".join(body_lines).strip()
        if body:
            sections.append(([t for _, t in stack], body))

    for line in text.split("\n"):
        match = HEADER_RE.match(line)
        if match:
            flush()
            body_lines.clear()
            level = len(match.group(1))
            title = match.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
        else:
            body_lines.append(line)
    flush()
    return sections


def split_oversized(path, body):
    """If a section's body is longer than MAX_WORDS, split it on paragraph
    breaks (blank lines) rather than an arbitrary character count, so we
    never cut a sentence in half."""
    word_count = len(body.split())
    if word_count <= MAX_WORDS:
        return [(path, body)]

    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    parts, current, current_words = [], [], 0
    for para in paragraphs:
        para_words = len(para.split())
        if current and current_words + para_words > MAX_WORDS:
            parts.append("\n\n".join(current))
            current, current_words = [], 0
        current.append(para)
        current_words += para_words
    if current:
        parts.append("\n\n".join(current))

    if len(parts) == 1:
        return [(path, parts[0])]
    return [(path + [f"part {i + 1} of {len(parts)}"], part) for i, part in enumerate(parts)]


def chunk_file(md_path: Path):
    text = md_path.read_text()
    sections = split_into_sections(text)

    chunks = []
    for path, body in sections:
        for sub_path, sub_body in split_oversized(path, body):
            chunks.append({"path": sub_path, "text": sub_body})
    return chunks


def main():
    all_chunks = []
    for md_path in sorted(SRC_DIR.glob("*.md")):
        file_chunks = chunk_file(md_path)
        for i, chunk in enumerate(file_chunks):
            all_chunks.append({
                "id": f"{md_path.stem}-{i:03d}",
                "text": chunk["text"],
                "metadata": {
                    "source_file": md_path.name,
                    "source_url": SOURCE_URLS[md_path.name],
                    "section": " > ".join(chunk["path"]),
                    "chunk_index": i,
                },
            })
        print(f"{md_path.name}: {len(file_chunks)} chunks")

    OUT_PATH.write_text(json.dumps(all_chunks, indent=2))
    print(f"\nTotal: {len(all_chunks)} chunks written to {OUT_PATH}")

    word_counts = [len(c["text"].split()) for c in all_chunks]
    print(f"Chunk size (words): min={min(word_counts)}, "
          f"max={max(word_counts)}, avg={sum(word_counts) / len(word_counts):.0f}")


if __name__ == "__main__":
    main()
