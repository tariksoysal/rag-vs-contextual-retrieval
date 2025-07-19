import gradio as gr
import faiss
import json
import numpy as np
import re
import time
from sentence_transformers import SentenceTransformer, util
from src.generation.generate_answers import generate_answer

# === Config ===
RAG_INDEX_PATH = 'data/processed/faiss_rag.index'
CTX_INDEX_PATH = 'data/processed/faiss_contextual.index'
DOCS_PATH = 'data/processed/rag_docs.jsonl'
CTX_DOCS_PATH = 'data/processed/contextual_docs.jsonl'
LOG_PATH = "evaluation_logs.jsonl"
EMBED_MODEL_NAME = 'sentence-transformers/all-MiniLM-L6-v2'
TOP_K = 5
SIM_THRESHOLD = 0.7
KEYWORD_THRESHOLD = 0.5
OLLAMA_MODEL = 'gemma3:latest'

# === Load model and indexes ===
model = SentenceTransformer(EMBED_MODEL_NAME)
rag_index = faiss.read_index(RAG_INDEX_PATH)
contextual_index = faiss.read_index(CTX_INDEX_PATH)

# === Load chunk metadata ===
with open(DOCS_PATH, 'r', encoding='utf-8') as f:
    rag_docs = [json.loads(line) for line in f]

with open(CTX_DOCS_PATH, 'r', encoding='utf-8') as f:
    ctx_docs = [json.loads(line) for line in f]

# === Preload gold answers (from prior eval logs) ===
gold_answers = {}
try:
    with open(LOG_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            entry = json.loads(line)
            if "gold_answer" in entry:
                gold_answers[entry["query"]] = entry["gold_answer"]
except FileNotFoundError:
    pass

# === Utility functions ===
def clean_html(text):
    return re.sub(r'<.*?>|&#xA;|&nbsp;|&quot;|&amp;', '', text)

def search_index(query, index, docs_list):
    query_vec = model.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_vec, TOP_K)
    results = [docs_list[i] for i in indices[0]]
    return results

def jaccard(a, b):
    a_set = set(a.lower().split())
    b_set = set(b.lower().split())
    return len(a_set & b_set) / len(a_set | b_set)

def run_eval(gold_answer, chunks):
    keyword_hits = 0
    gold_emb = model.encode(gold_answer, convert_to_tensor=True)
    chunk_embs = model.encode(chunks, convert_to_tensor=True)
    cos_scores = util.cos_sim(gold_emb, chunk_embs)[0]
    semantic_hits = sum(1 for score in cos_scores if score >= SIM_THRESHOLD)
    for chunk in chunks:
        if jaccard(gold_answer, chunk) >= KEYWORD_THRESHOLD:
            keyword_hits += 1
    return keyword_hits, semantic_hits

def run_rag(query, evaluate):
    gold = gold_answers.get(query)
    start = time.time()
    retrieved = search_index(query, rag_index, rag_docs)
    chunks = [clean_html(r.get("chunk", r.get("text", ""))) for r in retrieved]
    answer = generate_answer(query, retrieved, model=OLLAMA_MODEL)
    duration = round(time.time() - start, 2)
    sources = "\n\n".join([f"• {c[:300]}..." for c in chunks])
    metrics = ""
    if evaluate:
        if gold:
            k, s = run_eval(gold, chunks)
            metrics = f"✅ Semantic Recall@{TOP_K}: {s}/{TOP_K}\n🔎 Keyword Recall@{TOP_K}: {k}/{TOP_K}"
        else:
            metrics = "⚠️ No gold answer available for this query."
    answer += f"\n\n⏱️ Generated in {duration}s"
    return answer.strip(), sources, metrics

def run_contextual(query, evaluate):
    gold = gold_answers.get(query)
    start = time.time()
    retrieved = search_index(query, contextual_index, ctx_docs)
    chunks = [clean_html(r.get("chunk", r.get("text", ""))) for r in retrieved]
    answer = generate_answer(query, retrieved, model=OLLAMA_MODEL)
    duration = round(time.time() - start, 2)
    sources = "\n\n".join([f"• {c[:300]}..." for c in chunks])
    metrics = ""
    if evaluate:
        if gold:
            k, s = run_eval(gold, chunks)
            metrics = f"✅ Semantic Recall@{TOP_K}: {s}/{TOP_K}\n🔎 Keyword Recall@{TOP_K}: {k}/{TOP_K}"
        else:
            metrics = "⚠️ No gold answer available for this query."
    answer += f"\n\n⏱️ Generated in {duration}s"
    return answer.strip(), sources, metrics

# === Gradio UI ===
with gr.Blocks(title="RAG vs. Contextual Retrieval Comparison") as demo:
    gr.Markdown("""# RAG vs. Contextual Retrieval Comparison
Ask a question. RAG and Contextual pipelines will update independently as results come in.
""")
    with gr.Row():
        with gr.Column(scale=1):
            query = gr.Textbox(label="Enter your question", lines=1)
            evaluate = gr.Checkbox(label="Evaluate answers against gold", value=True)
            run_btn = gr.Button("Run Both")
            clear_btn = gr.Button("Clear")

        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    rag_out = gr.Textbox(label="🔵 RAG Answer")
                    rag_chunks = gr.Textbox(label="Top RAG Chunks", lines=6)
                    rag_eval = gr.Textbox(label="RAG Evaluation", lines=2)

                with gr.Column():
                    ctx_out = gr.Textbox(label="🔸 Contextual Answer")
                    ctx_chunks = gr.Textbox(label="Top Contextual Chunks", lines=6)
                    ctx_eval = gr.Textbox(label="Contextual Evaluation", lines=2)

    run_btn.click(fn=run_rag, inputs=[query, evaluate], outputs=[rag_out, rag_chunks, rag_eval])
    run_btn.click(fn=run_contextual, inputs=[query, evaluate], outputs=[ctx_out, ctx_chunks, ctx_eval])
    clear_btn.click(lambda: ("", "", "", "", "", ""), outputs=[rag_out, rag_chunks, rag_eval, ctx_out, ctx_chunks, ctx_eval])

if __name__ == "__main__":
    demo.launch()
