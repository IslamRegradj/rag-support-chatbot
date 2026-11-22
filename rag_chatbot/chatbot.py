"""The chatbot: load docs, index them, answer questions with sources."""

import os
from dataclasses import dataclass, field

from .chunker import chunk_documents
from .generator import NO_ANSWER, extractive_answer, llm_answer
from .loader import load_documents
from .retriever import TfidfRetriever


@dataclass
class Answer:
    text: str
    sources: list = field(default_factory=list)
    used_llm: bool = False


def _llm_available():
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


class RagChatbot:
    def __init__(self, docs_dir, chunk_size=120, overlap=30, top_k=3, use_llm=None):
        """`use_llm`: True/False to force a mode, None to enable it automatically
        when the `anthropic` package and an ANTHROPIC_API_KEY are present."""
        documents = load_documents(docs_dir)
        chunks = chunk_documents(documents, chunk_size, overlap)
        self.retriever = TfidfRetriever(chunks)
        self.top_k = top_k
        self.use_llm = _llm_available() if use_llm is None else use_llm

    def ask(self, question):
        hits = self.retriever.search(question, top_k=self.top_k)
        if not hits:
            return Answer(text=NO_ANSWER)

        sources = sorted({hit.chunk.source for hit in hits})
        if self.use_llm:
            try:
                return Answer(llm_answer(question, hits), sources, used_llm=True)
            except Exception:
                pass  # fall back to the offline answer below

        text, used_sources = extractive_answer(question, hits, self.retriever.vectorizer)
        return Answer(text, used_sources)
