import os
import time
import requests
import numpy as np
import streamlit as st
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================
# ENV / CONFIG
# ============================
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)

API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
if not API_KEY:
    st.error("Missing GEMINI_API_KEY. Add it to your .env file and restart.")
    st.stop()

HEADERS = {"Content-Type": "application/json", "x-goog-api-key": API_KEY}

EMBED_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:embedContent"
CHAT_URL  = "https://generativelanguage.googleapis.com/v1beta/models/gemini-flash-latest:generateContent"

KB_DIR = "knowledge_base"
os.makedirs(KB_DIR, exist_ok=True)

PROMPT_TEMPLATE = """
You are a helpful University of Ghana admissions assistant.

You will receive:
- A user question
- Optional context from official documents (may be empty)

Rules:
1) If the context is relevant, prioritize it and answer from it.
2) If context is empty or not useful, answer using general knowledge and best-practice guidance.
3) Be honest: if you're uncertain, say what the user should verify (e.g., UG admissions portal or admissions office).
4) Keep the answer concise (max 5 sentences). Use short bullet points if helpful.

When you use context, begin your answer with: "From the documents:"
When you do NOT use context, begin your answer with: "General guidance:"

Question: {question}
Context: {context}
Answer:
""".strip()

st.set_page_config(page_title="UG Admissions Assistant", page_icon="🎓", layout="centered")
st.title("🎓 UG Admissions Assistant")
st.caption("Ask questions about UG admissions. The assistant uses your backend documents when relevant.")


# ============================
# CHAT HISTORY (MULTI-CHAT)
# ============================
def _now_label():
    return datetime.now().strftime("%Y-%m-%d %H:%M")

def _new_chat():
    chat_id = f"chat_{int(time.time())}"
    st.session_state.chats[chat_id] = {
        "title": f"Chat {_now_label()}",
        "messages": [
            {"role": "assistant", "content": "Hi! Ask me anything about University of Ghana admissions."}
        ]
    }
    st.session_state.active_chat_id = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat_id" not in st.session_state or st.session_state.active_chat_id not in st.session_state.chats:
    _new_chat()

# Sidebar: list chats
with st.sidebar:
    st.subheader("💬 Chat History")

    if st.button("➕ New chat", use_container_width=True):
        _new_chat()

    # Show chats newest first
    chat_items = list(st.session_state.chats.items())
    chat_items.sort(key=lambda x: x[0], reverse=True)

    labels = [f"{v['title']}" for _, v in chat_items]
    ids = [k for k, _ in chat_items]

    selected = st.radio(
        "Select a chat",
        options=ids,
        format_func=lambda cid: st.session_state.chats[cid]["title"],
        label_visibility="collapsed"
    )
    st.session_state.active_chat_id = selected

    colA, colB = st.columns(2)
    with colA:
        if st.button("🧹 Clear", use_container_width=True):
            st.session_state.chats[selected]["messages"] = [
                {"role": "assistant", "content": "Chat cleared. Ask me anything about UG admissions."}
            ]
    with colB:
        if st.button("🗑 Delete", use_container_width=True):
            st.session_state.chats.pop(selected, None)
            if not st.session_state.chats:
                _new_chat()
            else:
                st.session_state.active_chat_id = list(st.session_state.chats.keys())[-1]
            st.rerun()

active_chat = st.session_state.chats[st.session_state.active_chat_id]


# ============================
# KB INDEXING (RAG)
# ============================
if "documents_store" not in st.session_state:
    st.session_state.documents_store = []  # list of {"content": str, "embedding": np.array}
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False
if "kb_signature" not in st.session_state:
    st.session_state.kb_signature = ""

def get_kb_pdf_paths():
    return sorted(
        os.path.join(KB_DIR, f)
        for f in os.listdir(KB_DIR)
        if f.lower().endswith(".pdf")
    )

def compute_signature(paths):
    parts = []
    for p in paths:
        try:
            parts.append(f"{os.path.basename(p)}:{os.path.getmtime(p)}")
        except Exception:
            parts.append(os.path.basename(p))
    return "|".join(parts)

def load_pdf(path: str):
    return PDFPlumberLoader(path).load()

def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True,
    )
    return splitter.split_documents(docs)

def get_embedding(text: str):
    text = (text or "").strip()
    if not text:
        return None
    text = text[:8000]

    body = {"content": {"parts": [{"text": text}]}}
    try:
        r = requests.post(EMBED_URL, headers=HEADERS, json=body, timeout=60)
        if r.status_code != 200:
            # show minimal message without debug blocks
            st.error("Embedding failed. Check your API key / quota.")
            return None

        data = r.json()
        values = data.get("embedding", {}).get("values")
        if not values:
            return None
        return np.array(values, dtype=np.float32)
    except:
        st.error("Embedding request failed. Check connection / API key.")
        return None

def generate_answer(question: str, context: str):
    prompt = PROMPT_TEMPLATE.format(question=question, context=context)
    body = {"contents": [{"parts": [{"text": prompt}]}]}
    try:
        r = requests.post(CHAT_URL, headers=HEADERS, json=body, timeout=60)
        if r.status_code != 200:
            st.error("Chat failed. Check your API key / quota.")
            return "Sorry — I couldn’t generate a response right now."

        data = r.json()
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except:
        st.error("Chat request failed. Check connection / API key.")
        return "Sorry — I couldn’t generate a response right now."

def index_chunks(chunks, delay_s: float = 0.35):
    st.session_state.documents_store = []
    for chunk in chunks:
        emb = get_embedding(chunk.page_content)
        if emb is not None:
            st.session_state.documents_store.append({"content": chunk.page_content, "embedding": emb})
        time.sleep(delay_s)

def retrieve(query: str, top_k: int = 3):
    q_emb = get_embedding(query)
    if q_emb is None or not st.session_state.documents_store:
        return []
    sims = []
    for doc in st.session_state.documents_store:
        sim = cosine_similarity([q_emb], [doc["embedding"]])[0][0]
        sims.append((sim, doc["content"]))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [txt for _, txt in sims[:top_k]]

def build_kb_if_needed():
    paths = get_kb_pdf_paths()
    if not paths:
        # no expander; just a simple notice
        st.warning("No PDFs found in knowledge_base. Add your UG admissions PDFs there.")
        return

    sig = compute_signature(paths)
    if st.session_state.kb_ready and st.session_state.kb_signature == sig:
        return

    st.session_state.kb_ready = False
    st.session_state.kb_signature = sig

    with st.spinner("Indexing knowledge base..."):
        all_docs = []
        for p in paths:
            all_docs.extend(load_pdf(p))
        chunks = split_docs(all_docs)
        index_chunks(chunks)

    st.session_state.kb_ready = True

# Build KB (quietly)
build_kb_if_needed()


# ============================
# MAIN CHAT UI
# ============================
# Render existing messages for the active chat
for msg in active_chat["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

# Input
question = st.chat_input("Ask a question about UG admissions")
if question:
    # save user message
    active_chat["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    # Retrieve context (may be empty)
    contexts = retrieve(question, top_k=3) if st.session_state.kb_ready else []
    context_text = "\n\n".join(contexts)

    # Generate answer
    answer = generate_answer(question, context_text)

    # save assistant message
    active_chat["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
