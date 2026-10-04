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

Latest results:

The evaluation set (`eval/eval_set.json`) has 20 answerable questions written from the real
text of the indexed documents (7 Constitution of India, 7 RBI AIFI master direction, 6 Income Tax
FAQs) plus 5 out-of-scope questions. A retrieved chunk counts as a hit only if it comes from the
expected file **and** contains the expected phrase. Every keyword was checked to exist in its file.

| Metric | Baseline (original defaults) | Tuned (current defaults) |
|---|---|---|
| Hit@5 | 80% (16/20) | **85% (17/20)** |
| MRR | 0.658 | **0.767** |
| Abstention accuracy (out-of-scope refused) | 40% (2/5) | **80% (4/5)** |
| Answerable questions answered (not refused) | 100% | 95% (19/20) |

Baseline: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=150`, `TOP_K=5`, `MIN_SIMILARITY=0.25`, no reranker.
Tuned: `CHUNK_SIZE=1000`, `CHUNK_OVERLAP=250`, `TOP_K=5`, `MIN_SIMILARITY=0.50`, `USE_RERANKER=true`.

Chunking and reranker sweep (`python -m eval.sweep`; each cell is Hit@5 / MRR):

| CHUNK_SIZE / OVERLAP | Chunks | No reranker | Reranker |
|---|---|---|---|
| 500 / 50 | 4688 | 0.70 / 0.546 | 0.85 / 0.658 |
| 500 / 100 | 5196 | 0.75 / 0.575 | 0.75 / 0.617 |
| 800 / 100 | 2999 | 0.75 / 0.654 | 0.85 / 0.760 |
| 800 / 200 | 3360 | 0.70 / 0.650 | 0.85 / 0.682 |
| 1000 / 150 | 2473 | 0.80 / 0.658 | 0.85 / 0.689 |
| **1000 / 250** | 2670 | 0.70 / 0.617 | **0.85 / 0.767** |
| 1500 / 200 | 1662 | 0.75 / 0.662 | 0.85 / 0.704 |

Other findings (1000 / 250 index):
- `TOP_K`: with the reranker, Hit@3 = 0.85, Hit@5 = 0.85, Hit@8 = 0.95. I kept 5 because the gain
  is one question out of 20 and 8 passages means a longer LLM prompt and more source cards.
- `MIN_SIMILARITY`: 0.50 refused 4/5 out-of-scope questions and wrongly refused 1/20 answerable
  ones. 0.60 refused 5/5 but wrongly refused 4/20, so I did not use it.

How to read these numbers:
- Only 20 + 5 questions, so one question moves a metric by 5 to 20 points. Treat differences of one
  question as noise. The settings were tuned on the same set that is reported, so the tuned
  numbers are optimistic.
- The evaluation measures retrieval and the similarity gate only. It does not call the LLM, so
  the LLM-side abstention and citation checks in `src/rag.py` are not part of these numbers.
- Known misses: Article 21, the CET1 minimum and ITR-1 (SAHAJ) were not retrieved in the top 5;
  "What is Form-16?" is retrieved but refused by the similarity gate (best similarity 0.39);
  "What is the punishment for theft under the Indian Penal Code?" is not refused (similarity 0.56):
  a vector-similarity gate cannot reliably separate out-of-scope questions that are close in topic
  to the indexed legal text.
  Article 21 is not a Hindi/bilingual extraction problem (the extracted text is English only).
  Its chunk shares 982 characters with Articles 20, 21A and 22, and the text says "21. Protection
  of life..." without the word "Article", so a question containing "Article 21" has little to match.
  Questions phrased like the text ("protection of life and personal liberty") do retrieve it.
  Cutting chunks at numbered-provision boundaries improved its rank (64th to 9th) but left Hit@5
  at 85% and moved MRR from 0.767 to 0.792, within noise on 20 questions, so I did not keep it.

Tuning loop: set one variable at a time via `.env` (`MIN_SIMILARITY`, `TOP_K`,
`USE_RERANKER`), or `CHUNK_SIZE`/`CHUNK_OVERLAP` followed by `python -m src.ingest`, then
re-run `python -m eval.sweep` / `python -m eval.run_eval`. The MiniLM embedder truncates input
at ~256 tokens (about 1000 characters), so `CHUNK_SIZE` above ~1000 does not help the vector side.

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
