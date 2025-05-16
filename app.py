import gradio as gr
import faiss
import json
import numpy as np
import re
from sentence_transformers import SentenceTransformer
from src.generation.generate_answers import generate_answer

# === Config ===
RAG_INDEX_PATH = 'data/processed/faiss_rag.index'
CTX_INDEX_PATH = 'data/processed/faiss_contextual.index'
RAG_DOCS_PATH = 'data/processed/rag_docs.jsonl'
CTX_DOCS_PATH = 'data/processed/contextual_docs.jsonl'
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5
OLLAMA_MODEL = 'gemma3:latest'

# === Helpers ===
def clean_html(text):
    text = re.sub(r'<.*?>', '', text)                  # remove HTML tags
    text = re.sub(r'&#x[A-Fa-f0-9]+;', ' ', text)      # remove entities like &#xA;
    text = re.sub(r'&nbsp;|&quot;|&amp;', ' ', text)   # common HTML entities
    text = text.replace('\n', ' ').replace('\r', ' ')  # newlines
    text = re.sub(r'\s+', ' ', text)                   # normalize whitespace
    return text.strip()

def search_index(query, index, docs):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, TOP_K)
    return [docs[i] for i in indices[0]]

# === Load Models and Data ===
model = SentenceTransformer(EMBED_MODEL_NAME)
rag_index = faiss.read_index(RAG_INDEX_PATH)

try:
    contextual_index = faiss.read_index(CTX_INDEX_PATH)
    contextual_ready = True
except:
    contextual_index = None
    contextual_ready = False

with open(RAG_DOCS_PATH, 'r', encoding='utf-8') as f:
    rag_docs = [json.loads(line) for line in f]

with open(CTX_DOCS_PATH, 'r', encoding='utf-8') as f:
    contextual_docs = [json.loads(line) for line in f]

# === RAG Pipeline ===
def rag_pipeline(query, mode):
    if mode == "Contextual Retrieval" and contextual_ready:
        retrieved = search_index(query, contextual_index, contextual_docs)
    else:
        retrieved = search_index(query, rag_index, rag_docs)

    answer = generate_answer(query, retrieved, model=OLLAMA_MODEL)
    sources = "\n\n".join([f"• {clean_html(r['text'])[:200]}..." for r in retrieved])
    return answer.strip(), sources

# === Gradio UI ===
iface = gr.Interface(
    fn=rag_pipeline,
    inputs=[
        gr.Textbox(label="Enter your question"),
        gr.Radio(["RAG", "Contextual Retrieval"], label="Retrieval Mode", value="RAG")
    ],
    outputs=[
        gr.Textbox(label="Generated Answer"),
        gr.Textbox(label="Top Retrieved Chunks", lines=10),
    ],
    title="RAG Demo with Ollama",
    description="Ask a technical question and choose between standard RAG and Contextual Retrieval.",
)

if __name__ == "__main__":
    iface.launch()

