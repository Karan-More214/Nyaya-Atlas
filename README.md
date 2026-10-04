---
title: NyayaAtlas
emoji: ⚖️
colorFrom: indigo
colorTo: blue
sdk: streamlit
app_file: app.py
pinned: false
---

# ⚖️ NyayaAtlas

**Navigate Indian law and policy, with exact source citations.**

NyayaAtlas is a citation-first RAG (Retrieval-Augmented Generation) assistant. You ask a
question about Indian law or policy (Constitution, RBI circulars, Income Tax FAQs, or any
PDFs you add) and it answers using only those documents, showing the **document and page**
for every claim. If the documents don't contain the answer, it says so instead of guessing.

> For information only. Not legal advice.

## Features

- **Page-level citations:** every chunk is tied to one page, so sources are exact.
- **Hybrid retrieval:** BM25 keyword search + vector search, merged with Reciprocal Rank Fusion.
- **Optional reranker:** cross-encoder for better ordering (`USE_RERANKER=true`).
- **Abstention guardrails:** weak retrieval or an insufficient-context answer returns "not found".
- **Works without an API key:** shows the top passages with page numbers (extractive mode).
- **Evaluation script:** Hit@k, MRR and abstention accuracy on your own question set.

## Architecture

**Phase 1: indexing** (`python -m src.ingest`)

```
PDFs in data/raw/
   -> read page by page (pypdf), clean the text, skip blank/scanned pages
   -> cut each page into ~1000-character chunks (150 overlap)
   -> tag every chunk with: file, title, page number
   -> turn each chunk into a vector (MiniLM embedding model, runs locally)
   -> store vectors in ChromaDB, and the raw text in chunks.jsonl (for keyword search)
```

**Phase 2: answering** (every question)

```
Question
   -> meaning search (vectors in ChromaDB)  -> top 20
   -> keyword search (BM25)                 -> top 20
   -> merge both lists (Reciprocal Rank Fusion), optional reranker
   -> keep the top 5 passages
   -> gate: is the best vector match similar enough? if not, "I could not find this
      in the indexed documents."
   -> send the 5 numbered passages + question to the LLM
   -> LLM answers only from them, with [1], [2] citations
   -> check: answer must contain valid citations, otherwise abstain
   -> UI shows the answer plus source cards (document, page, text)
```

Without an API key, the LLM steps are skipped and the app shows the top passages with
their file and page.

## Project structure

```
nyayaatlas/
├── app.py              # Streamlit UI
├── requirements.txt
├── .env.example
├── data/raw/           # put your PDFs here
├── eval/
│   ├── eval_set.json   # your test questions
│   └── run_eval.py
└── src/
    ├── config.py
    ├── ingest.py       # build the index
    ├── retriever.py    # hybrid search
    └── rag.py          # answer generation + guardrails
```

## Quick start (VS Code)

1. Open the `nyayaatlas` folder in VS Code.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # Windows:  .venv\Scripts\activate
   # Mac/Linux: source .venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Add documents (PDF/TXT) to `data/raw/` (see `data/README.md`).
5. *(Optional)* Copy `.env.example` to `.env` and add your `ANTHROPIC_API_KEY` for full answers.
6. Build the index:
   ```bash
   python -m src.ingest
   ```
7. Run the app:
   ```bash
   streamlit run app.py
   ```

## Evaluation

Edit `eval/eval_set.json` to match your documents, then run:

```bash
python -m eval.run_eval
```

Report the numbers (Hit@5, MRR, abstention accuracy) in your LinkedIn post and here:

> **Status:** no source PDFs were present in `data/raw/` when this was last updated, so the
> numbers below have not been measured yet. `eval/eval_set.json` is still the 4-question
> starter set. Add your PDFs, write ~20 answerable + ~5 out-of-scope questions, run the
> command above and paste the real results here.

| Metric | Result |
|---|---|
| Hit@5 | _not yet measured_ |
| MRR | _not yet measured_ |
| Abstention accuracy | _not yet measured_ |

Tuning loop: set one variable at a time via `.env` (`MIN_SIMILARITY`, `TOP_K`,
`USE_RERANKER`), or `CHUNK_SIZE`/`CHUNK_OVERLAP` followed by `python -m src.ingest`, then
re-run the evaluation. Note that the MiniLM embedder truncates input at ~256 tokens
(about 1000 characters), so `CHUNK_SIZE` above ~1000 will not help the vector side.

## Deploy a live demo (free)

**Hugging Face Spaces (Streamlit):**
1. Create a new Space, choose the Streamlit SDK. This README's front matter already
   sets `sdk: streamlit` and `app_file: app.py`.
2. Push the repo to the Space (`git remote add space https://huggingface.co/spaces/<user>/<space>`).
3. Add the PDFs you are allowed to redistribute under `data/raw/` **in the Space repo only**
   (they are gitignored here; use `git add -f data/raw/<file>.pdf` on a Space-only branch,
   and Git LFS for files over 10 MB).
4. Add `ANTHROPIC_API_KEY` under Settings > Secrets (optional).
5. No manual indexing: on first start `app.py` builds the index from `data/raw/` if
   `index/` is missing, then loads it. If there are no documents the app shows a clear message.

**Streamlit Community Cloud:** push the repo to GitHub, connect it at share.streamlit.io,
and add your key under Secrets. The index is built automatically on first start.

Only publish documents you are allowed to redistribute.

## Tuning

| Setting | Effect |
|---|---|
| `MIN_SIMILARITY` | Higher = more "not found" answers, fewer hallucinations |
| `TOP_K` | Passages sent to the LLM |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | Granularity of citations |
| `USE_RERANKER` | Better ranking, slower and larger download |

## Limitations

- Scanned PDFs need OCR first (ingest reports how many pages it skipped).
- Citations use the physical PDF page number, which can differ from the page number printed on the page.
- `.txt` files have no pages and are cited as page 1.
- Tables in PDFs may extract poorly.
- Laws change; the answers are only as current as the documents you index.

## License

MIT
