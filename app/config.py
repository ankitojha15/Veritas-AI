# This file holds ALL settings in one place.
# Think of it like a remote control for our project.
# If you want to change a model, just change it here.
import os
# --- Folders ---
PDF_FOLDER = "data/pdfs"          # put your PDF files here
PROCESSED_FOLDER = "data/processed"  # we save clean text here
INDEX_FOLDER = "data/index"       # we save search index here

# --- Chunking (cutting text into small pieces) ---
# Parent = big piece (for reading), Child = small piece (for searching)
PARENT_SIZE = 1000   # big piece = 1000 characters
CHILD_SIZE = 300     # small piece = 300 characters
OVERLAP = 50         # overlap so we don't lose meaning

# --- Models (all free and small) ---
# Dense model = understands meaning
DENSE_MODEL = "all-MiniLM-L6-v2"
# Rerank model = picks best answer
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
# LLM model = writes final answer (Groq is free)
# Old llama-3.3 was shut down Aug 2026, new one is gpt-oss
LLM_MODEL = "openai/gpt-oss-120b"

# --- Search settings ---
TOP_K = 5  # how many pieces to pick for final answer

QDRANT_URL = os.getenv("QDRANT_URL", "")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
QDRANT_COLLECTION = "veritas"
DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"