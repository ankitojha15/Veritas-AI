# takes question, gives answer.

from fastapi import FastAPI
from pydantic import BaseModel
from app.indexing import build_all
from app.generation import answer_question

app = FastAPI(title="VeritasAI")


class AskIn(BaseModel):
    question: str  # what user asks


class AskOut(BaseModel):
    answer: str
    citations: list = []
    no_answer: bool = False

class IndexOut(BaseModel):
    done: bool = True
    pages: int = 0
    chunks: int = 0


@app.get("/")
def home():
    return {"msg": "VeritasAI is running. Use /ask to ask."}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/index" , response_model=IndexOut)
def make_index():
    # Read PDFs and make search files.
    result = build_all()
    return {"done": True,"pages":result["pages"],"chunks":result["chunks"]}


@app.post("/ask", response_model=AskOut)
def ask(data: AskIn):
    # Find answer with pages.
    out = answer_question(data.question)
    return {
        "answer": out["answer"],
        "citations": out.get("citations", []),
        "no_answer": out.get("no_answer", False),
    }
