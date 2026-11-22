"""Find the chunks most relevant to a question (TF-IDF + cosine similarity)."""

from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from .chunker import Chunk


@dataclass(frozen=True)
class Hit:
    chunk: Chunk
    score: float


class TfidfRetriever:
    def __init__(self, chunks):
        if not chunks:
            raise ValueError("The retriever needs at least one chunk")
        self.chunks = list(chunks)
        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2), stop_words="english", sublinear_tf=True
        )
        self.matrix = self.vectorizer.fit_transform(c.text for c in self.chunks)

    def search(self, query, top_k=3, min_score=0.05):
        """Return up to `top_k` chunks, best first, ignoring weak matches."""
        query_vector = self.vectorizer.transform([query])
        scores = linear_kernel(query_vector, self.matrix).ravel()
        best_first = scores.argsort()[::-1][:top_k]
        return [
            Hit(self.chunks[i], float(scores[i]))
            for i in best_first
            if scores[i] >= min_score
        ]
