# Step 4: Two indexes - Qdrant for meaning, BM25 for exact words.

import os
import pickle
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.retrievers import BM25Retriever
from app import config

load_dotenv()


def get_embedding_model():
    return HuggingFaceEndpointEmbeddings(
        model=config.DENSE_MODEL,
        huggingfacehub_api_token=config.HF_TOKEN
    )


def build_dense(child_docs):
    model = get_embedding_model()
    client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
    client.recreate_collection(
        collection_name=config.QDRANT_COLLECTION,
        vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    )
    texts = [c.page_content for c in child_docs]
    vectors = model.embed_documents(texts)
    points = []
    for i, (c, v) in enumerate(zip(child_docs, vectors)):
        points.append(PointStruct(
            id=i,
            vector=v,
            payload={
                "text": c.page_content,
                "page": c.metadata["page"],
                "source": c.metadata["source"],
                "parent_text": c.metadata.get("parent_text", c.page_content),
                "parent_id": c.metadata.get("parent_id", ""),
                "chunk_id": c.metadata.get("chunk_id", f"child-{i}")
            }
        ))
    client.upsert(collection_name=config.QDRANT_COLLECTION, points=points)
    print(f"Saved dense to Qdrant {config.QDRANT_COLLECTION} count {len(points)}")
    return client


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
    import shutil
    from app.chunking import chunk_and_save

    parents, childs = chunk_and_save()
    if len(childs) == 0:
        print("No chunks. Add PDFs first.")
        try:
            client = QdrantClient(url=config.QDRANT_URL, api_key=config.QDRANT_API_KEY)
            client.delete_collection(collection_name=config.QDRANT_COLLECTION)
        except Exception:
            pass
        for p in [os.path.join(config.INDEX_FOLDER, "bm25.pkl")]:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        return {"dense": None, "bm25": None, "pages": 0, "chunks": 0}
    dense = build_dense(childs)
    bm25 = build_bm25(childs)
    print("Both indexes done.")
    return {"dense": dense, "bm25": bm25, "pages": len(parents), "chunks": len(childs)}


if __name__ == "__main__":
    build_all()