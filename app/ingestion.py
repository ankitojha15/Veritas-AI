# Like asking LangChain to open a book and copy each page.

import os
import json
from langchain_community.document_loaders import PyMuPDFLoader
from app import config


def load_one_pdf(pdf_path):
    # Load 1 PDF
    # Returns a list of Documents.
    # Each Document has: page_content (text) + metadata (page, source)
    loader = PyMuPDFLoader(pdf_path)  # LangChain book opener
    docs = loader.load()  # read all pages

    # Fix page number:starts from 0, we want from 1
    for d in docs:
        d.metadata["page"] = d.metadata["page"] + 1
        d.metadata["source"] = os.path.basename(pdf_path)

    # Remove empty pages
    clean_docs = []
    for d in docs:
        if d.page_content.strip() != "":
            clean_docs = clean_docs + [d]
    return clean_docs


def load_all_pdfs():
    # Load ALL PDFs from data/pdfs folder.
    # Save to data/processed/all_pages.json for next steps.
    all_docs = []

    files = os.listdir(config.PDF_FOLDER)
    for f in files:
        if f.endswith(".pdf"):
            full_path = os.path.join(config.PDF_FOLDER, f)
            docs = load_one_pdf(full_path)
            all_docs = all_docs + docs
            print(f"Read {f}: {len(docs)} pages")

    # Save simple copy as json
    os.makedirs(config.PROCESSED_FOLDER, exist_ok=True)
    out_path = os.path.join(config.PROCESSED_FOLDER, "all_pages.json")
    simple_list = []
    for d in all_docs:
        simple_list.append({
            "text": d.page_content,
            "page": d.metadata["page"],
            "source": d.metadata["source"],
        })
    with open(out_path, "w") as out:
        json.dump(simple_list, out, indent=2)

    print(f"Saved {len(all_docs)} pages to {out_path}")
    return all_docs


# If we run this file, load all PDFs
if __name__ == "__main__":
    load_all_pdfs()
