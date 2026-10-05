# Step 4: Two indexes - FAISS for meaning, BM25 for exact words.

import os
import pickle
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from app import config

load_dotenv()  # read .env file (for Huggingface + Groq keys)


def get_embedding_model():
    name = config.DENSE_MODEL
    if "/" not in name:  # add full name if short
        name = "sentence-transformers/" + name
    return HuggingFaceEmbeddings(model_name=name)


def build_dense(child_docs):
    model = get_embedding_model()
    shop = FAISS.from_documents(child_docs, model)
    os.makedirs(config.INDEX_FOLDER, exist_ok=True)
    path = os.path.join(config.INDEX_FOLDER, "faiss")
    shop.save_local(path)
    print(f"Saved dense index to {path}")
    return shop


def build_bm25(child_docs):
    tool = BM25Retriever.from_documents(child_docs)
    tool.k = config.TOP_K
    os.makedirs(config.INDEX_FOLDER, exist_ok=True)
    path = os.path.join(config.INDEX_FOLDER, "bm25.pkl")
    with open(path, "wb") as f:
        pickle.dump(tool, f)
    print(f"Saved BM25 index to {path}")
    return tool


def build_all():
    # Full job: cut text, then make both indexes.
    # If new PDF has 0 pages, wipe old index so we never answer from old PDF.
    import shutil
    from app.chunking import chunk_and_save

    parents, childs = chunk_and_save()
    if len(childs) == 0:
        print("No chunks. Add PDFs first.")
        for p in [os.path.join(config.INDEX_FOLDER, "faiss"),
                  os.path.join(config.INDEX_FOLDER, "bm25.pkl")]:
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p)
                elif os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        return {"dense": None, "bm25": None, "pages": 0, "chunks": 0}
    dense = build_dense(childs)
    bm25 = build_bm25(childs)
    print("Both indexes done.")
    return {"dense":dense, "bm25":bm25 , "pages":len(parents), "chunks":len(childs)}


# If we run this file, build both indexes
if __name__ == "__main__":
    build_all()
