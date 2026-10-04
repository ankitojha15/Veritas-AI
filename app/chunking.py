# Parent = big piece (for context). Child = small piece (for searching).

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app import config
import json
import os


def make_chunks(docs):
    # Take pages (Documents) and make parent + child pieces.
    # Returns: parent_docs, child_docs
    parent_cut = RecursiveCharacterTextSplitter(
        chunk_size=config.PARENT_SIZE,
        chunk_overlap=config.OVERLAP,
    )
    child_cut = RecursiveCharacterTextSplitter(
        chunk_size=config.CHILD_SIZE,
        chunk_overlap=config.OVERLAP,
    )

    parent_docs = parent_cut.split_documents(docs)

    # Give each parent a simple id: parent-0, parent-1...
    for i, p in enumerate(parent_docs):
        p.metadata["chunk_id"] = f"parent-{i}"
        p.metadata["type"] = "parent"

    # Now make childs from each parent
    child_docs = []
    num = 0
    for p in parent_docs:
        small_pieces = child_cut.split_text(p.page_content)
        for piece in small_pieces:
            c = Document(
                page_content=piece,
                metadata={
                    "page": p.metadata["page"],
                    "source": p.metadata["source"],
                    "type": "child",
                    "chunk_id": f"child-{num}",
                    "parent_id": p.metadata["chunk_id"],
                    "parent_text": p.page_content,  # keep big text with us
                },
            )
            child_docs.append(c)
            num = num + 1

    return parent_docs, child_docs


def chunk_and_save():
    # Load pages, cut them, save to json.
    from app.ingestion import load_all_pdfs

    docs = load_all_pdfs()
    if len(docs) == 0:
        print("No pages found. Put PDFs in data/pdfs first.")
        return [], []

    parents, childs = make_chunks(docs)

    # Save simple copy so next steps can use it
    os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_FOLDER, "chunks.json")
    simple_list = []
    for c in childs:
        simple_list.append({
            "text": c.page_content,
            "page": c.metadata["page"],
            "source": c.metadata["source"],
            "chunk_id": c.metadata["chunk_id"],
            "parent_id": c.metadata["parent_id"],
            "parent_text": c.metadata["parent_text"],
        })
    with open(out_path, "w") as f:
        json.dump(simple_list, f, indent=2)

    print(f"Made {len(parents)} parents and {len(childs)} childs.")
    print(f"Saved to {out_path}")
    return parents, childs

if __name__ == "__main__":
    chunk_and_save()
