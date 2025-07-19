This directory holds generated retrieval artifacts.
Large FAISS indexes and the JSONL files describing document chunks are not stored in the repository.
Run the retriever scripts to populate this folder:

```
python src/retrieval/rag_retriever.py
python src/retrieval/contextual_retriever.py
```

The scripts will create `faiss_rag.index`, `faiss_contextual.index`, `rag_docs.jsonl`, and `contextual_docs.jsonl`.

- `faiss_rag.index`: Contains the vector embeddings for retrieval-augmented generation (RAG) retrieval.
- `faiss_contextual.index`: Contains the vector embeddings for contextual retrieval.
- `rag_docs.jsonl`: Stores metadata and chunked text data for documents used in RAG retrieval.
- `contextual_docs.jsonl`: Stores metadata and chunked text data for documents used in contextual retrieval.
