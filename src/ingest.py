"""Build the NyayaAtlas index.

Usage:
    python -m src.ingest

Reads every PDF / TXT in data/raw, splits it page by page into chunks (so each
chunk carries an exact page number), embeds the chunks and stores them in
ChromaDB. A plain-text copy of the chunks is also written for BM25 keyword search.
"""
import json
import re
import shutil
from pathlib import Path

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from src import config


def clean(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"-\n(?=\w)", "", text)  # join words hyphenated across lines
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Split on paragraph/sentence boundaries where possible, with overlap."""
    if len(text) <= size:
        return [text] if text else []
    chunks, start = [], 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            window = text[start:end]
            cut = max(window.rfind("\n\n"), window.rfind(". "), window.rfind("\n"))
            if cut > size * 0.5:
                end = start + cut + 1
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start = max(end - overlap, start + 1)
    return chunks


def pretty_title(path: Path) -> str:
    return path.stem.replace("_", " ").replace("-", " ").strip().title()


MIN_PAGE_CHARS = 30  # pages with less text than this are treated as blank / scanned


def load_pages(path: Path, stats: dict | None = None):
    """Yield (page_number, text) for a document. Page numbers are the 1-based
    physical PDF page index (what a PDF viewer shows), never a guess.

    `stats` (optional) is filled with total / skipped page counts."""
    stats = stats if stats is not None else {}
    stats.update(total=0, skipped=0)
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            try:
                reader.decrypt("")  # many "encrypted" PDFs just have an empty user password
            except Exception:
                pass
        n_pages = len(reader.pages)
        stats["total"] = n_pages
        for i in range(n_pages):
            try:
                text = clean(reader.pages[i].extract_text() or "")
            except Exception:
                text = ""
            if len(text) >= MIN_PAGE_CHARS:
                yield i + 1, text
            else:
                stats["skipped"] += 1
    elif path.suffix.lower() == ".txt":
        text = clean(path.read_text(encoding="utf-8", errors="ignore"))
        stats["total"] = 1
        if text:
            yield 1, text
        else:
            stats["skipped"] = 1


def main():
    files = sorted(
        p for p in config.DATA_DIR.glob("**/*") if p.suffix.lower() in {".pdf", ".txt"}
    )
    if not files:
        raise SystemExit(
            f"No PDF/TXT files found in {config.DATA_DIR}. "
            "Add your documents there (see data/README.md) and run again."
        )

    chunks = []
    for path in files:
        title = pretty_title(path)
        rel = path.relative_to(config.DATA_DIR).as_posix()  # unique even across subfolders
        n_before = len(chunks)
        stats: dict = {}
        try:
            pages = list(load_pages(path, stats))
        except Exception as e:  # corrupt / unreadable file: skip it, keep going
            print(f"WARNING: could not read {rel}: {e}. Skipping.")
            continue
        for page_no, text in tqdm(pages, desc=path.name, leave=False):
            for j, piece in enumerate(
                split_text(text, config.CHUNK_SIZE, config.CHUNK_OVERLAP)
            ):
                chunks.append(
                    {
                        "id": f"{rel}::p{page_no}::c{j}",
                        "text": piece,
                        "source": rel,
                        "title": title,
                        "page": page_no,
                    }
                )
        msg = f"{rel}: {len(chunks) - n_before} chunks from {len(pages)}/{stats['total']} pages"
        if stats["skipped"]:
            msg += f" ({stats['skipped']} blank/scanned pages skipped - OCR them to include)"
        print(msg)

    if not chunks:
        raise SystemExit("No extractable text found. Scanned PDFs need OCR first.")

    # Fresh index each run
    if config.INDEX_DIR.exists():
        shutil.rmtree(config.INDEX_DIR)
    config.INDEX_DIR.mkdir(parents=True)

    with open(config.CHUNKS_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Embedding {len(chunks)} chunks with {config.EMBED_MODEL} ...")
    model = SentenceTransformer(config.EMBED_MODEL)
    client = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    collection = client.create_collection(
        config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    batch = 128
    for i in tqdm(range(0, len(chunks), batch), desc="Indexing"):
        part = chunks[i : i + batch]
        embeddings = model.encode(
            [c["text"] for c in part], normalize_embeddings=True
        ).tolist()
        collection.add(
            ids=[c["id"] for c in part],
            documents=[c["text"] for c in part],
            embeddings=embeddings,
            metadatas=[
                {"source": c["source"], "title": c["title"], "page": c["page"]}
                for c in part
            ],
        )
    print(f"Done. Index saved to {config.INDEX_DIR}")


if __name__ == "__main__":
    main()
