---
title: Veritas AI
emoji: 📚
sdk: streamlit
sdk_version: 1.63.0
app_file: app/ui_app.py
pinned: false
---

# Veritas AI v1 — Ask your PDFs, get answers with proof

Live demo: https://knowledgebaseai-pzi7.streamlit.app/

I built this because I was tired of LLMs making up answers. So I made a simple RAG bot that only answers from your PDFs, and shows `[Page X]` with every fact. If it's not in the PDF, it says so clearly.

Veritas means truth in Latin. That's the whole idea here.

## What it does

You upload a PDF, hit Build Index, and ask questions. You get short answers with page numbers. No hallucinations, no extra gyaan.

Example:
> Q: What do cats like?
> A: Cats like milk [Page 1]

If you ask something out of syllabus like "capital of Mars?", it replies: `I don't know from the PDFs.` That's intentional.

## How I built it

1. **Load PDFs** - Used `PyMuPDFLoader` from LangChain. Page by page text + fixed page numbers (LangChain starts from 0, I do +1)
2. **Chunking** - `RecursiveCharacterTextSplitter` with Parent (1000 chars) + Child (300 chars). Search happens on small chunks, answer uses big parent text. Overlap 50 so sentences don't break.
3. **Indexing** - Two indexes: `FAISS + all-MiniLM-L6-v2` for meaning search, `BM25` for exact word search
4. **Retrieval** - Hybrid search with `EnsembleRetriever (RRF, c=60)` + rerank with `cross-encoder/ms-marco-MiniLM-L-6-v2`. Top 5 chunks go to LLM.
5. **Generation** - LangGraph pipeline with 4 steps: rewrite question -> retrieve -> write answer -> verify citations with regex. Model is `openai/gpt-oss-120b` via Groq. Temperature 0.

Everything is LangChain-only. No LlamaIndex, no vector DB service. Runs locally.

## Run it yourself

1. Put PDFs in `data/pdfs/` or use Upload button in UI. Copy `.env.example` to `.env` and add:
   - `GROQ_API_KEY` - for answers
   - `HUGGINGFACE_API_KEY` - optional, embeddings model is public
2. Build index: `./aienv/bin/python -m app.indexing`
3. Ask:
   - UI local: `./aienv/bin/python -m streamlit run app/ui_app.py`
   - API: `./aienv/bin/python -m uvicorn app.main_api:app --reload` -> POST `/ask` with `{"question": "..."}`

## Files

- `app/config.py` - all settings in one place
- `app/ingestion.py` - read PDFs
- `app/chunking.py` - parent/child split
- `app/indexing.py` - FAISS + BM25 build
- `app/retrieval.py` - hybrid + RRF + rerank
- `app/generation.py` - LangGraph: rewrite -> retrieve -> answer -> verify
- `app/main_api.py` - FastAPI (`/ask`, `/index`, `/health`)
- `app/ui_app.py` - Streamlit chat UI
- `evals/golden.json` - 60 test cases (50 normal + 10 adversarial)
- `tests/test_all.py` - end-to-end sanity tests

## What I learned / Limitations

- Text PDFs work great. Scanned image PDFs don't - that needs OCR, planning for v2.
- BM25 saved me for exact keywords, dense saved me for meaning. You need both.
- Citation check with regex is simple but works - catches most hallucinations.

## Next: Veritas Sentinel v2

v1 retrieves and answers. v2 will correct itself - retrieval grader + query rewrite loop + web search fallback (CRAG). Work in progress.