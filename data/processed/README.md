This directory holds generated retrieval artifacts.
Large FAISS indexes and the JSONL files describing document chunks are not stored in the repository.
Run the retriever scripts to populate this folder:

```
python src/retrieval/rag_retriever.py
python src/retrieval/contextual_retriever.py
```

The scripts will create `faiss_rag.index`, `faiss_contextual.index`, `rag_docs.jsonl`, and `contextual_docs.jsonl`.

