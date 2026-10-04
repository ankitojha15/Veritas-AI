# Step 6: Write answer using LangChain + LangGraph.
# Graph: fix question -> find pages -> write answer -> check pages.
# Like a small team passing one paper from hand to hand.

import re
from typing import TypedDict
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END
from app import config

load_dotenv()


class State(TypedDict):
    # One paper passed between steps.
    question: str
    rewrite: str
    docs: list
    context: str
    answer: str
    citations: list
    no_answer: bool


def get_llm():
    # Smart writer (Groq is free).
    return ChatGroq(model=config.LLM_MODEL, temperature=0)


def rewrite_query(query):
    # Make question better for search. If no key, keep original.
    try:
        llm = get_llm()
        msg = f"Rewrite this question for book search. Keep meaning. Return only new question. Question: {query}"
        out = llm.invoke(msg)
        return out.content.strip()
    except Exception:
        return query


def build_context(docs):
    # Join docs into one text with page tags. Use big parent text.
    # Also removes same parent twice.
    seen = [] #holds used parent IDs. It prevents duplicates.
    parts = [] #holds text blocks
    for d in docs:
        pid = d.metadata.get("parent_id", d.metadata.get("chunk_id"))
        if pid in seen:
            continue
        seen.append(pid)
        text = d.metadata.get("parent_text", d.page_content)
        page = d.metadata["page"]
        source = d.metadata["source"]
        parts.append(f"[Page {page} from {source}]\n{text}")
    return "\n\n".join(parts)


def verify_citations(answer, docs):
    # Check: every [Page X] in answer must be in our docs.
    pages_in_docs = []
    for d in docs:
        pages_in_docs.append(str(d.metadata["page"]))
    pages_in_answer = re.findall(r"\[Page (\d+)", answer)
    good = []
    bad = []
    for p in pages_in_answer:
        if p in pages_in_docs:
            good.append(p)
        else:
            bad.append(p)
    return {"good": good, "bad": bad, "ok": len(bad) == 0}


# --- Graph steps (each takes the paper, adds one line) ---

def fix_q(state: State):
    # Step 1: fix question.
    return {"rewrite": rewrite_query(state["question"])}


def find(state: State):
    # Step 2: find pages.
    from app.retrieval import search
    docs = search(state["rewrite"])
    return {"docs": docs}


def write(state: State):
    # Step 3: write answer from pages.
    if len(state["docs"]) == 0:
        return {"answer": "I didn't get it from the PDF.", "context": "", "no_answer": True, "citations": []}
    context = build_context(state["docs"])
    llm = get_llm()
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You answer only from context. Give short answer. Add [Page X] after each fact.Answer in bullet points. If not in context, say: I didn't get it from the PDF."),
        ("human", "Question: {q}\n\nContext:\n{ctx}"),
    ])
    chain = prompt | llm
    out = chain.invoke({"q": state["question"], "ctx": context})
    return {"context": context, "answer": out.content.strip()}


def check(state: State):
    # Step 4: check pages + no-answer.
    if state.get("no_answer"):
        return {}
    if "I didn't get it from the PDF." in state["answer"]:
        return {"no_answer": True, "citations": []}
    ok = verify_citations(state["answer"], state["docs"])
    cites = []
    for d in state["docs"]:
        cites.append({"page": d.metadata["page"], "source": d.metadata["source"]})
    only_good = []
    for c in cites:
        if str(c["page"]) in ok["good"]:
            if c not in only_good:
                only_good.append(c)
    return {"citations": only_good, "no_answer": False}


def build_graph():
    # Join steps into a line: fix -> find -> write -> check -> end.
    g = StateGraph(State)
    g.add_node("fix_q", fix_q)
    g.add_node("find", find)
    g.add_node("write", write)
    g.add_node("check", check)
    g.set_entry_point("fix_q")
    g.add_edge("fix_q", "find")
    g.add_edge("find", "write")
    g.add_edge("write", "check")
    g.add_edge("check", END)
    return g.compile()


_graph = None


def answer_question(query):
    # Full job through the graph.
    global _graph
    if _graph is None:
        _graph = build_graph()
    out = _graph.invoke({"question": query})
    return {
        "answer": out.get("answer", "I didn't get it from the PDF."),
        "citations": out.get("citations", []),
        "no_answer": out.get("no_answer", False),
        "rewrite": out.get("rewrite", query),
    }


if __name__ == "__main__":
    from langchain_core.documents import Document
    fake = [Document(page_content="Cats like milk", metadata={"page": 1, "source": "a.pdf", "chunk_id": "c0", "parent_id": "p0", "parent_text": "Cats like milk"})]
    print(build_context(fake))
    print(verify_citations("Cats drink [Page 1]", fake))
