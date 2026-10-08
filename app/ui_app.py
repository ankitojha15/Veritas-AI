# Pretty chat screen using Streamlit + LangChain.
# Run with: ./aienv/bin/python -m streamlit run app/ui_app.py

import streamlit as st
import os
import sys
import time
import requests

# Make "app" work both locally and on Streamlit Cloud
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from app import config
from app.indexing import build_all
from app.generation import answer_question


def stream_words(text):
    # Fake streaming: give one word at a time, like typing.
    for w in text.split():
        yield w + " "
        time.sleep(0.02)

# Look
st.set_page_config(page_title="VeritasAI", page_icon="📚", layout="wide")

# Left side: steps (stays still)
with st.sidebar:
    st.title("📚 VeritasAI")
    st.write("**1. Upload PDFs**")
    files = st.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)
    if files:
        os.makedirs(config.PDF_FOLDER, exist_ok=True)
        # Auto-remove old PDFs: only new upload stays
        for old in os.listdir(config.PDF_FOLDER):
            if old.endswith(".pdf"):
                try:
                    os.remove(os.path.join(config.PDF_FOLDER, old))
                except Exception:
                    pass
        for f in files:
            path = os.path.join(config.PDF_FOLDER, f.name)
            with open(path, "wb") as out:
                out.write(f.getbuffer())
        st.success(f"Saved {len(files)} PDF(s).")
        st.warning("👉 Next step: click 🔨 Build Index below! You must do this after every upload.")
    st.write("**2. Build Index**")
    if st.button("🔨 Build Index", use_container_width=True):
        with st.spinner("Reading..."):
            r = requests.post(f"{config.API_URL}/index", timeout=300)
            data = r.json()
        st.session_state.chats = []
        if not childs:
            st.error("No text found in this PDF. It may be scanned images. Try a text PDF.")
        else:
            st.success("Ready! Old PDF removed. Working on new PDF only. Ask on the right.")
    st.divider()
    st.caption("Answers always show [Page X].")

# Right side: chat (ask box always at bottom)
st.header("💬 Ask your PDFs")

# Remember old chats
if "chats" not in st.session_state:
    st.session_state.chats = []

# Show old chats first (no streaming here, just plain text)
for chat in st.session_state.chats:
    with st.chat_message("user"):
        st.write(chat["q"])
    with st.chat_message("assistant"):
        st.write(chat["a"])
        if chat["cites"]:
            st.caption("📄 " + ", ".join([f"Page {c['page']} ({c['source']})" for c in chat["cites"]]))

# Ask box at the very bottom (always below answers)
q = st.chat_input("Type your question here...")
if q:
    with st.chat_message("user"):
        st.write(q)
    with st.spinner("Thinking..."):
        out = answer_question(q)
    with st.chat_message("assistant"):
        st.write_stream(stream_words(out["answer"]))
        if out.get("citations"):
            st.caption("📄 " + ", ".join([f"Page {c['page']} ({c['source']})" for c in out["citations"]]))
        if out.get("no_answer"):
            st.warning("No answer in PDFs.")
    # Save so it stays on screen, next ask box comes below
    st.session_state.chats.append({"q": q, "a": out["answer"], "cites": out.get("citations", [])})
