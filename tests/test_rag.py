from pathlib import Path

import pytest

from rag_chatbot import RagChatbot
from rag_chatbot.chunker import chunk_text
from rag_chatbot.generator import NO_ANSWER

DOCS = Path(__file__).resolve().parent.parent / "docs"


def test_chunks_overlap_and_cover_the_text():
    text = " ".join(f"w{i}" for i in range(100))
    chunks = chunk_text("a.md", text, size=40, overlap=10)
    assert chunks[0].text.split()[-10:] == chunks[1].text.split()[:10]
    assert chunks[-1].text.split()[-1] == "w99"


def test_short_text_gives_one_chunk():
    assert len(chunk_text("a.md", "just a few words", size=40, overlap=10)) == 1


def test_empty_text_gives_no_chunk():
    assert chunk_text("a.md", "   ") == []


@pytest.mark.parametrize("size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_chunk_settings_are_rejected(size, overlap):
    with pytest.raises(ValueError):
        chunk_text("a.md", "some text", size=size, overlap=overlap)


@pytest.fixture(scope="module")
def bot():
    return RagChatbot(DOCS, use_llm=False)


def test_refund_question_uses_the_refund_document(bot):
    answer = bot.ask("How long do I have to ask for a refund?")
    assert "support_refunds.md" in answer.sources
    assert "30 days" in answer.text


def test_password_question_uses_the_access_document(bot):
    answer = bot.ask("How do I reset my password?")
    assert "support_account_access.md" in answer.sources


def test_leave_question_uses_the_hr_document(bot):
    answer = bot.ask("How many days of annual leave do I get?")
    assert "hr_leave_policy.md" in answer.sources
    assert "25" in answer.text


def test_unrelated_question_gets_a_polite_no_answer(bot):
    answer = bot.ask("What is the capital of Australia?")
    assert answer.text == NO_ANSWER
    assert answer.sources == []


def test_missing_folder_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        RagChatbot(tmp_path / "does-not-exist", use_llm=False)
