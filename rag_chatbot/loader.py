"""Load text documents from a folder."""

from dataclasses import dataclass
from pathlib import Path

SUPPORTED_SUFFIXES = {".txt", ".md"}


@dataclass(frozen=True)
class Document:
    source: str  # path relative to the documentation folder
    text: str


def _headings_to_sentences(text):
    """Turn Markdown headings into their own sentence so they stay separate
    from the first sentence of the section once newlines are flattened."""
    lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            stripped = stripped.lstrip("#").strip()
            if stripped and stripped[-1] not in ".!?":
                stripped += "."
        lines.append(stripped)
    return "\n".join(lines)


def load_documents(folder):
    """Read every .txt and .md file under `folder` (recursively)."""
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"Documentation folder not found: {root}")

    documents = []
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES:
            text = _headings_to_sentences(path.read_text(encoding="utf-8")).strip()
            if text:
                documents.append(Document(source=path.relative_to(root).as_posix(), text=text))

    if not documents:
        raise ValueError(f"No .txt or .md documents found in: {root}")
    return documents
