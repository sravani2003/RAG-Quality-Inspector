import numpy as np
import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

class TfidfFaissStore:
    def __init__(self, chunks: list[str]):
        self.chunks = chunks
        self.vectorizer = TfidfVectorizer(stop_words="english", max_features=8000)

        X = self.vectorizer.fit_transform(chunks).astype(np.float32).toarray()
        X = self._l2_normalize(X)

        self.index = faiss.IndexFlatIP(X.shape[1])  # cosine via normalized inner product
        self.index.add(X)

    def _l2_normalize(self, X, eps=1e-10):
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        return X / (norms + eps)

    def search(self, query: str, k=4):
        q = self.vectorizer.transform([query]).astype(np.float32).toarray()
        q = self._l2_normalize(q)
        scores, idxs = self.index.search(q, k)

        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

def build_vector_store(chunks: list[str]):
    return TfidfFaissStore(chunks)

def retrieve(query: str, store: TfidfFaissStore, k=4):
    return store.search(query, k=k)
