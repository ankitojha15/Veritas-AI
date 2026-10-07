# Hybrid = EnsembleRetriever mixes dense (meaning) + BM25 (words) with RRF.
# Rerank = ask a smart checker to pick the best.

import os
import pickle
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from langchain_classic.retrievers import EnsembleRetriever
from app import config
from app.indexing import get_embedding_model

load_dotenv()


from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from app import config
from app.indexing import get_embedding_model

def load_dense():
    model = get_embedding_model()
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    store = QdrantVectorStore(
        client=client,
        collection_name=config.QDRANT_COLLECTION,
        embedding=model
    )
    return store


def load_bm25():
    # Load saved BM25 index
    path = os.path.join(config.INDEX_FOLDER, "bm25.pkl")
    with open(path, "rb") as f:
        tool = pickle.load(f)
    return tool


def hybrid_search(query, dense_shop, bm25_tool):
    # Mix dense + BM25 with LangChain RRF. c=60 is the RRF stable number.
    n = config.TOP_K * 2
    dense_box = dense_shop.as_retriever(search_kwargs={"k": n})
    team = EnsembleRetriever(retrievers=[dense_box, bm25_tool], weights=[0.5, 0.5], c=60)
    return team.invoke(query)


_checker = None  # keep the judge in memory so we load only once


def rerank(query, docs, top_k=5):
    # Ask cross-encoder: how well does each doc match query?
    global _checker
    if _checker is None:
        _checker = HuggingFaceCrossEncoder(model_name=config.RERANK_MODEL)
    pairs = []
    for d in docs:
        pairs.append((query, d.page_content))
    marks = _checker.score(pairs)

    # Join docs with marks and sort
    together = []
    for i in range(len(docs)):
        together.append((docs[i], marks[i]))
    together.sort(key=lambda x: x[1], reverse=True)

    best = []
    for doc, mark in together[:top_k]:
        best.append(doc)
    return best


def search(query):
    # Full search: hybrid (RRF) + rerank. Returns best docs.
    dense_shop = load_dense()
    bm25_tool = load_bm25()
    mixed = hybrid_search(query, dense_shop, bm25_tool)
    final = rerank(query, mixed, top_k=config.TOP_K)
    return final


# If we run this file, ask a test question
if __name__ == "__main__":
    answers = search("test question")
    for a in answers:
        print(a.page_content[:200], "| page:", a.metadata["page"])
