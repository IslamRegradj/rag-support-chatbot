"""Split documents into overlapping chunks of words."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    source: str
    index: int
    text: str


def chunk_text(source, text, size=120, overlap=30):
    """Cut `text` into windows of `size` words that share `overlap` words.

    The overlap keeps a sentence that falls on a boundary readable in at
    least one of the two chunks.
    """
    if size <= 0:
        raise ValueError("size must be positive")
    if not 0 <= overlap < size:
        raise ValueError("overlap must be between 0 and size - 1")

    words = text.split()
    step = size - overlap
    chunks = []
    for index, start in enumerate(range(0, len(words), step)):
        window = words[start : start + size]
        chunks.append(Chunk(source=source, index=index, text=" ".join(window)))
        if start + size >= len(words):
            break
    return chunks


def chunk_documents(documents, size=120, overlap=30):
    chunks = []
    for document in documents:
        chunks.extend(chunk_text(document.source, document.text, size, overlap))
    return chunks
