"""Turn retrieved chunks into an answer.

Two modes:
- extractive (default, works offline): return the sentences that best match the question
- LLM (optional): ask a language model to answer using only the retrieved text
"""

import re

NO_ANSWER = (
    "I could not find this in the documentation. "
    "Please contact the support or HR team."
)

DEFAULT_LLM_MODEL = "claude-sonnet-5-5"


def _split_sentences(text):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


def extractive_answer(question, hits, vectorizer, max_sentences=2):
    """Pick the sentences that cover the most important words of the question.

    Each sentence is scored by the share of the question's TF-IDF weight it
    contains. Ties go to the longer sentence, then to document order.
    Returns (answer_text, sources_used).
    """
    if not hits:
        return NO_ANSWER, []

    analyzer = vectorizer.build_analyzer()
    idf = dict(zip(vectorizer.get_feature_names_out(), vectorizer.idf_))
    query_terms = {term for term in analyzer(question) if term in idf}
    if not query_terms:
        return NO_ANSWER, []
    total_weight = sum(idf[term] for term in query_terms)

    candidates = []  # (sentence, source)
    seen = set()
    for hit in hits:
        for sentence in _split_sentences(hit.chunk.text):
            if sentence not in seen:
                seen.add(sentence)
                candidates.append((sentence, hit.chunk.source))

    scores = []
    for sentence, _ in candidates:
        shared = query_terms & set(analyzer(sentence))
        scores.append(sum(idf[term] for term in shared) / total_weight)

    # On equal scores, prefer the longer (more informative) sentence, then document order.
    best = sorted(
        (i for i, s in enumerate(scores) if s > 0),
        key=lambda i: (-scores[i], -len(candidates[i][0]), i),
    )
    chosen = sorted(best[:max_sentences])  # read them in their original order
    if not chosen:
        return NO_ANSWER, []

    text = " ".join(candidates[i][0] for i in chosen)
    sources = sorted({candidates[i][1] for i in chosen})
    return text, sources


def llm_answer(question, hits, model=DEFAULT_LLM_MODEL):
    """Ask a language model to answer from the retrieved context only."""
    import anthropic  # optional dependency, imported only when used

    context = "\n\n".join(f"[{h.chunk.source}]\n{h.chunk.text}" for h in hits)
    prompt = (
        "Answer the question using only the documentation excerpts below. "
        f'If the answer is not there, reply exactly: "{NO_ANSWER}"\n\n'
        f"Documentation:\n{context}\n\nQuestion: {question}"
    )
    client = anthropic.Anthropic()
    message = client.messages.create(
        model=model,
        max_tokens=500,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text.strip()
