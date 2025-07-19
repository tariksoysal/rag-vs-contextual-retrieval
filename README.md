# rag-vs-contextual-retrieval

This project compares two retrieval strategies for LLM question answering:
Retrieval-Augmented Generation (RAG) versus a "Contextual" approach where
document chunks are summarized before indexing.  The demo in `app.py` lets you
query both pipelines side by side.

## Setup

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare the training data and FAISS indexes. The scripts below expect your
   raw posts in `data/processed/combined.jsonl` (not included in this repo):
   ```bash
   # Split into train/eval before chunking
   python src/data_preparation/split_questions_before_chunking.py

   # Create text chunks for the RAG baseline
   python src/data_preparation/chunk_documents.py

   # Optionally generate summarized chunks for contextual retrieval
   python src/data_preparation/chunk_documents_contextual.py

   # Split the chunked data into train and eval sets
   python src/data_preparation/split_eval_set.py

   # Build retrieval indexes
   python src/retrieval/rag_retriever.py
   python src/retrieval/contextual_retriever.py
   ```

## Running the retrieval demo

Launch the Gradio interface to compare RAG and contextual retrieval:
```bash
python app.py
```
A browser window will open where you can ask questions and inspect the top
retrieved passages.

## Example commands

```bash
# Retrieve from a RAG index in a terminal
python test_rag.py

# Retrieve from a contextual index in a terminal
python test_contextual.py

# Evaluate recall metrics on all questions
python src/evaluation/eval_rag_all.py
python src/evaluation/eval_contextual_all.py
```
