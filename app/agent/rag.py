"""
RAG over unstructured school documents (circulars, handbook, fee policy, ...).

WHY RAG AND NOT JUST TOOLS?
---------------------------
Tools answer questions the *database* knows: attendance, marks, fees.
RAG answers questions only *prose* knows: "what is the leave policy?",
"what's the uniform rule on Saturdays?", "how do I apply for a bus route change?"

You need both. The agent decides which by picking a tool -- `search_school_documents`
is just one more tool in the list.

THE PIPELINE
------------
  ingest:  document -> chunks -> embedding vectors -> Chroma (on disk)
  query:   question -> embedding vector -> cosine nearest chunks -> into the prompt

EMBEDDINGS: we use Chroma's *built-in* default, `all-MiniLM-L6-v2` running on
ONNX Runtime. It is free, offline, 384-dimensional, and -- crucially -- does not
drag in PyTorch (~2GB). Good enough for a school corpus. If you later want
better recall, swap in `bge-small-en-v1.5` via sentence-transformers.

CHUNKING -- the part everyone gets wrong
----------------------------------------
Naive fixed-size chunking measurably hurts recall. Our first version split the
attendance policy into two 800-char blobs, and the query "how do I apply for
leave" retrieved the paragraph about lunch-break marking instead of the section
literally titled "Applying for leave".

Two fixes, both worth internalising:

1. SPLIT ON STRUCTURE FIRST. Markdown headings are authored semantic boundaries
   -- respect them, and only size-split a section that is genuinely too long.
2. PREPEND A BREADCRUMB. Each chunk starts with
   "Attendance Policy > Applying for leave". The heading words then live inside
   the embedded vector, so a question phrased like the heading actually matches
   it -- and the LLM can cite the section, not just the file.

Overlap still matters for the size-split path: without it, a sentence straddling
a boundary is retrievable from neither chunk.
"""
from __future__ import annotations

import os
import re

import chromadb

from config import Config

_client = None
_collection = None


def get_collection():
    global _client, _collection
    if _collection is None:
        os.makedirs(Config.CHROMA_DIR, exist_ok=True)
        _client = chromadb.PersistentClient(path=Config.CHROMA_DIR)
        _collection = _client.get_or_create_collection(
            name=Config.RAG_COLLECTION,
            metadata={"hnsw:space": "cosine"},   # cosine is the right metric for MiniLM
        )
    return _collection


HEADING = re.compile(r"^(#{1,6})\s+(.*)$", re.MULTILINE)


def _split_by_size(body: str, size: int, overlap: int) -> list[str]:
    """Paragraph-aware size split with overlap. Only used on oversized sections."""
    paras = [p.strip() for p in body.split("\n\n") if p.strip()]
    out, buf = [], ""
    for p in paras:
        if not buf:
            buf = p
        elif len(buf) + len(p) + 2 <= size:
            buf = f"{buf}\n\n{p}"
        else:
            out.append(buf)
            tail = buf[-overlap:]
            buf = f"{tail}\n\n{p}"
    if buf:
        out.append(buf)
    return out


def chunk_text(text: str, size: int = 700, overlap: int = 120) -> list[str]:
    """
    Heading-aware chunking. Returns chunks that each begin with a breadcrumb like

        Attendance Policy > Applying for leave

    so the section title is part of what gets embedded.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())

    # Walk the headings, carrying a stack of ancestor titles.
    sections: list[tuple[str, str]] = []      # (breadcrumb, body)
    stack: list[tuple[int, str]] = []
    pos, current = 0, None

    for m in HEADING.finditer(text):
        if current is not None:
            sections.append((current, text[pos:m.start()].strip()))
        level, title = len(m.group(1)), m.group(2).strip()
        while stack and stack[-1][0] >= level:
            stack.pop()
        stack.append((level, title))
        current = " > ".join(t for _, t in stack)
        pos = m.end()

    if current is not None:
        sections.append((current, text[pos:].strip()))
    else:
        sections.append(("", text))           # plain text / PDF with no headings

    chunks: list[str] = []
    for crumb, body in sections:
        if not body:
            continue
        prefix = f"{crumb}\n\n" if crumb else ""
        if len(body) <= size:
            chunks.append(prefix + body)
        else:
            chunks.extend(prefix + part for part in _split_by_size(body, size, overlap))
    return chunks


def _read_file(path: str) -> str:
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader

        return "\n\n".join((page.extract_text() or "") for page in PdfReader(path).pages)
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.read()


def ingest_directory(directory: str | None = None) -> dict:
    """Re-index every .md/.txt/.pdf in data/docs. Idempotent: wipes then rebuilds."""
    directory = directory or Config.DOCS_DIR
    col = get_collection()

    # simplest correct strategy for a small corpus: full rebuild
    existing = col.get(include=[])["ids"]
    if existing:
        col.delete(ids=existing)

    n_docs = n_chunks = 0
    for fname in sorted(os.listdir(directory)):
        if not fname.lower().endswith((".md", ".txt", ".pdf")):
            continue
        path = os.path.join(directory, fname)
        chunks = chunk_text(_read_file(path))
        if not chunks:
            continue
        col.add(
            ids=[f"{fname}::{i}" for i in range(len(chunks))],
            documents=chunks,
            metadatas=[{"source": fname, "chunk": i} for i in range(len(chunks))],
        )
        n_docs += 1
        n_chunks += len(chunks)

    return {"documents": n_docs, "chunks": n_chunks, "path": Config.CHROMA_DIR}


def search(query: str, k: int | None = None) -> list[dict]:
    """Return the k nearest chunks with a similarity score and their source file."""
    col = get_collection()
    if col.count() == 0:
        return []

    k = min(k or Config.RAG_TOP_K, col.count())
    res = col.query(query_texts=[query], n_results=k)

    out = []
    for doc, meta, dist in zip(
        res["documents"][0], res["metadatas"][0], res["distances"][0]
    ):
        out.append(
            {
                "source": meta.get("source"),
                "similarity": round(1 - dist, 3),   # cosine distance -> similarity
                "text": doc,
            }
        )
    return out
