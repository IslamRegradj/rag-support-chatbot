"""Command-line chat: python -m rag_chatbot.cli --docs docs"""

import argparse

from .chatbot import RagChatbot


def main():
    parser = argparse.ArgumentParser(description="Ask questions about your documentation.")
    parser.add_argument("--docs", default="docs", help="folder with .txt / .md files")
    parser.add_argument("--no-llm", action="store_true", help="never call a language model")
    args = parser.parse_args()

    bot = RagChatbot(args.docs, use_llm=False if args.no_llm else None)
    mode = "language model" if bot.use_llm else "extractive (offline)"
    print(f"Documentation loaded. Answer mode: {mode}. Type 'quit' to leave.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if question.lower() in {"quit", "exit", "q"}:
            break
        if not question:
            continue
        answer = bot.ask(question)
        print(f"Bot: {answer.text}")
        if answer.sources:
            print(f"     Sources: {', '.join(answer.sources)}")
        print()


if __name__ == "__main__":
    main()
