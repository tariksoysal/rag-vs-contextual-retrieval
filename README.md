# rag-vs-contextual-retrieval
Prototype comparing Retrieval-Augmented Generation and Contextual Retrieval for LLM-based QA

## License

This project is licensed under the [MIT License](LICENSE).

## Generating retrieval indexes
The FAISS index and metadata files used by the examples are not committed to Git. Generate them by running:

```bash
python src/retrieval/rag_retriever.py
python src/retrieval/contextual_retriever.py
```

This will produce the `.index` and `.jsonl` files inside `data/processed/`.

