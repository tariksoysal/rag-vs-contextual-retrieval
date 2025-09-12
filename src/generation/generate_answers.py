"""Utilities to format prompts and query an Ollama model for answers."""

import json
import subprocess

import re
import html

def clean_html(text):
    """Remove HTML tags and decode common entities."""
    if not isinstance(text, str):
        return ""
    no_tags = re.sub(r"<[^>]*>", "", text)
    return html.unescape(no_tags)

def format_prompt(context_chunks, query):
    """Construct a simple question‑answer prompt from retrieved chunks."""
    context_text = "\n\n".join([clean_html(c['text']) for c in context_chunks])
    return f"""Answer the following question using the context below.

Question:
{query}

Context:
{context_text}

Answer:"""


def run_ollama_prompt(prompt, model="gemma3:latest"):
    """Send ``prompt`` to the local Ollama model and return its response."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode("utf-8"),
        capture_output=True
    )
    return result.stdout.decode("utf-8")

def generate_answer(query, retrieved_chunks, model="gemma3:latest"):
    """Generate an answer for ``query`` given ``retrieved_chunks``."""
    prompt = format_prompt(retrieved_chunks, query)
    answer = run_ollama_prompt(prompt, model=model)
    return answer

# example call
if __name__ == "__main__":
    # Load dummy chunks from previous RAG result to test
    with open("data/processed/rag_docs.jsonl", "r") as f:
        docs = [json.loads(f.readline()) for _ in range(5)]

    query = "Why is quicksort faster in practice?"
    response = generate_answer(query, docs, model="gemma3:latest")
    print(response)
