# rag-support-chatbot

A small retrieval-augmented (RAG) chatbot that answers customer-support or HR questions using a company's internal documentation, and tells you which document each answer came from.

The repository includes three fictional sample documents (leave policy, refunds, account access) so you can try it right away. Replace them with your own `.txt` or `.md` files.

## How it works

1. **Load**: reads every `.txt` and `.md` file from a folder.
2. **Chunk**: splits each document into overlapping windows of words, so a sentence on a boundary is not lost.
3. **Retrieve**: indexes the chunks with TF-IDF (unigrams and bigrams) and finds the ones closest to the question with cosine similarity. Weak matches are discarded.
4. **Answer**:
   - *Offline mode (default)*: returns the sentences that cover the most important words of the question.
   - *LLM mode (optional)*: sends the retrieved chunks to a language model and asks it to answer from them only.
5. If nothing relevant is found, the bot says so instead of guessing.

## Quick start

```bash
git clone https://github.com/IslamRegradj/rag-support-chatbot.git
cd rag-support-chatbot
pip install -r requirements.txt
python -m rag_chatbot.cli --docs docs
```

Example session (offline mode):

```text
You: How many days of annual leave do I get?
Bot: Every full-time employee receives 25 days of paid annual leave per calendar year. Unused annual leave can be carried over to the next year, up to a maximum of 5 days.
     Sources: hr_leave_policy.md

You: When do I need a medical certificate?
Bot: A medical certificate is required when the absence lasts more than three consecutive days.
     Sources: hr_leave_policy.md

You: What is the capital of Australia?
Bot: I could not find this in the documentation. Please contact the support or HR team.
```

### Use it from Python

```python
from rag_chatbot import RagChatbot

bot = RagChatbot("docs", use_llm=False)
answer = bot.ask("How do I reset my password?")
print(answer.text, answer.sources)
```

### Optional: LLM mode

Install the Anthropic SDK and set an API key. The chatbot then switches to LLM mode automatically (use `--no-llm` to force offline mode).

```bash
pip install anthropic
export ANTHROPIC_API_KEY=your_key_here
python -m rag_chatbot.cli --docs docs
```

If the model call fails, the chatbot falls back to the offline answer.

## Tests

```bash
python -m pytest
```

The tests cover chunking (overlap, empty and invalid input), retrieval of the right document for each sample question, the "no answer" case, and a missing documentation folder.

## Project structure

```text
rag_chatbot/
  loader.py       read documents from a folder
  chunker.py      split documents into overlapping chunks
  retriever.py    TF-IDF index and cosine-similarity search
  generator.py    offline answer and optional LLM answer
  chatbot.py      RagChatbot: ties everything together
  cli.py          command-line chat
docs/             sample documents (fictional)
tests/            pytest tests
```

## Limitations and next steps

- Retrieval is lexical (TF-IDF): it matches words, not meaning, so a question phrased with different words than the document can be missed. Replacing it with embeddings and a vector database is the natural next step.
- Only `.txt` and `.md` files are supported. PDF and Word loaders are not included yet.
- The offline answer returns existing sentences and cannot combine information from several places the way a language model can.
- No web interface yet. A small FastAPI endpoint or a Streamlit page would be an easy addition.
