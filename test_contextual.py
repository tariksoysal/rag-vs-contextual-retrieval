import pytest
import numpy as np

faiss = pytest.importorskip("faiss")


def build_test_index():
    docs = [{"chunk": f"chunk {i}"} for i in range(5)]
    embeddings = np.stack([np.array([float(i), float(i)], dtype="float32") for i in range(5)])
    index = faiss.IndexFlatL2(2)
    index.add(embeddings)
    return index, docs, embeddings


def search_index(vec, index, docs, k):
    _, idx = index.search(np.array([vec], dtype="float32"), k)
    return [docs[i] for i in idx[0]]


def test_contextual_retrieval():
    index, docs, embeddings = build_test_index()
    results = search_index(embeddings[0], index, docs, k=3)
    assert len(results) == 3
    assert all("chunk" in r for r in results)
