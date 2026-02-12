import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from google import genai
from PIL import Image

# ============================
# ENV / CONFIG
# ============================
dotenv_path = find_dotenv()
load_dotenv(dotenv_path)
API_KEY = os.environ.get("GEMINI_API_KEY")

if not API_KEY:
    st.error("API key not found! Please set GEMINI_API_KEY in your environment.")
    st.stop()

client = genai.Client(api_key=API_KEY)

# ============================
# PROMPT TEMPLATE
# ============================
PROMPT_TEMPLATE = """
You are a helpful University of Ghana admissions assistant.

You will receive:
- A user question
- Optional context from official documents (may be empty)

Rules:
1) If the context is relevant, prioritize it and answer from it.
2) If context is empty or not useful, answer using general knowledge.
3) If unsure, advise user to verify on the official UG admissions portal.
4) Keep answers concise (max 5 sentences).

When using context, begin with: "From the documents:"
When not using context, begin with: "General guidance:"

Question: {question}
Context: {context}
Answer:
""".strip()

st.set_page_config(page_title="UG Admissions Assistant", page_icon="🎓")
st.title("🎓 UG Admissions Assistant")
st.caption("Ask questions about UG admissions or upload your results to check cut-offs.")
st.write("Loaded key starts with:", API_KEY[:8])

# ============================
# CHAT STATE
# ============================
def new_chat():
    chat_id = f"chat_{int(time.time())}"
    st.session_state.chats[chat_id] = {
        "title": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "messages": [
            {"role": "assistant", "content": "Hi! Ask me anything about UG admissions or upload your results to check cut-offs."}
        ]
    }
    st.session_state.active_chat = chat_id

if "chats" not in st.session_state:
    st.session_state.chats = {}
if "active_chat" not in st.session_state:
    new_chat()

with st.sidebar:
    st.subheader("💬 Chats")
    if st.button("➕ New Chat"):
        new_chat()
    for cid in reversed(list(st.session_state.chats.keys())):
        if st.button(st.session_state.chats[cid]["title"], key=cid):
            st.session_state.active_chat = cid

active_chat = st.session_state.chats[st.session_state.active_chat]

# ============================
# EMBEDDINGS
# ============================
def get_embedding(text: str):
    text = text.strip()
    if not text:
        return None
    try:
        result = client.models.embed_content(
            model="gemini-embedding-001",
            contents=text
        )
        embedding_vector = result.embeddings[0].values
        return np.array(embedding_vector, dtype=np.float32)
    except Exception as e:
        st.error(f"Embedding error: {e}")
        return None

# ============================
# GENERATE RESPONSE
# ============================
def generate_answer(question: str, context: str):
    try:
        prompt = PROMPT_TEMPLATE.format(
            question=question,
            context=context
        )
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        st.error(f"Chat error: {e}")
        return "Sorry, I couldn’t generate a response."

# ============================
# KNOWLEDGE BASE (RAG)
# ============================
if "documents" not in st.session_state:
    st.session_state.documents = []
if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

def build_kb():
    KB_DIR = "./knowledge_base"
    if not os.path.exists(KB_DIR):
        st.warning(f"PDF folder '{KB_DIR}' not found.")
        return

    pdf_files = [
        os.path.join(KB_DIR, f)
        for f in os.listdir(KB_DIR)
        if f.lower().endswith(".pdf")
    ]

    if not pdf_files:
        st.warning("No PDFs found in knowledge base folder.")
        return

    st.session_state.documents = []
    with st.spinner("Indexing PDFs..."):
        all_docs = []
        for pdf in pdf_files:
            loader = PDFPlumberLoader(pdf)
            all_docs.extend(loader.load())
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = splitter.split_documents(all_docs)
        for chunk in chunks:
            emb = get_embedding(chunk.page_content)
            if emb is not None:
                st.session_state.documents.append({
                    "content": chunk.page_content,
                    "embedding": emb
                })
            time.sleep(0.3)
    st.session_state.kb_ready = True

if not st.session_state.kb_ready:
    build_kb()

def retrieve(query: str, top_k: int = 3):
    if not st.session_state.documents:
        return []
    q_emb = get_embedding(query)
    if q_emb is None:
        return []
    sims = []
    for doc in st.session_state.documents:
        sim = cosine_similarity([q_emb], [doc["embedding"]])[0][0]
        sims.append((sim, doc["content"]))
    sims.sort(key=lambda x: x[0], reverse=True)
    return [text for _, text in sims[:top_k]]

# ============================
# CUT-OFF CHECK
# ============================
cut_offs = {
    "Computer Science": 60,
    "Law": 55,
    "Medicine": 70,
    "Engineering": 65
}

def extract_results(file):
    temp_path = f"temp_uploaded_{int(time.time())}_{file.name}"
    with open(temp_path, "wb") as f:
        f.write(file.read())

    try:
        if file.type == "application/pdf":
            loader = PDFPlumberLoader(temp_path)
            docs = loader.load()
            return "\n".join([d.page_content for d in docs])
        elif file.type == "text/csv":
            df = pd.read_csv(temp_path)
            return df.to_dict(orient="records")
        elif file.type in [
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            "application/vnd.ms-excel"
        ]:
            df = pd.read_excel(temp_path)
            return df.to_dict(orient="records")
        elif file.type.startswith("image/"):
            return f"Image file '{file.name}' uploaded successfully."
        else:
            return None
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

def check_cut_off(results):
    feedback = []
    if isinstance(results, str):  # PDF text or image notice
        feedback.append(results if results.startswith("Image") else
                        "PDF uploaded. Please enter your program and score as text for cut-off checking.")
    else:
        for record in results:
            program = record.get("Program") or record.get("course")
            score = float(record.get("Score") or record.get("score", 0))
            if program in cut_offs:
                if score >= cut_offs[program]:
                    feedback.append(f"You meet the cut-off for {program} ({score} >= {cut_offs[program]}).")
                else:
                    feedback.append(f"You do NOT meet the cut-off for {program} ({score} < {cut_offs[program]}).")
    return "\n".join(feedback)

# ============================
# CHAT UI
# ============================
for msg in active_chat["messages"]:
    st.chat_message(msg["role"]).write(msg["content"])

uploaded_file = st.file_uploader(
    "Upload your results (PDF, CSV, Excel, or Image)", 
    type=["pdf", "csv", "xlsx", "xls", "png", "jpg", "jpeg"]
)

if uploaded_file:
    st.chat_message("user").write(f"Uploaded file: {uploaded_file.name}")
    results = extract_results(uploaded_file)
    cut_off_feedback = check_cut_off(results)
    st.chat_message("assistant").write(cut_off_feedback)

question = st.chat_input("Ask a question about UG admissions")

if question:
    active_chat["messages"].append({"role": "user", "content": question})
    st.chat_message("user").write(question)

    contexts = retrieve(question) if st.session_state.kb_ready else []
    context_text = "\n\n".join(contexts)

    answer = generate_answer(question, context_text)

    active_chat["messages"].append({"role": "assistant", "content": answer})
    st.chat_message("assistant").write(answer)
