"""Simplify article text into a short spoken English bulletin."""

from __future__ import annotations

from pipeline.llm import chat_complete, is_configured


SYSTEM_PROMPT = """/no_think
You are rewriting a news article to be read aloud as a radio bulletin.
Rules:
- Maximum 80 words
- First sentence: state the single most important fact
- Use simple spoken language, no jargon, no passive voice
- Third sentence onwards: explain briefly
- Final sentence: "What this means for you:" followed by one practical implication
- Output plain text only, no markdown"""


def summarise(article_text: str) -> str:
    text = article_text.strip()
    if not text:
        return "There is not enough article text to make a reliable spoken bulletin."
    if not is_configured():
        return _local_demo_summary(text)

    return chat_complete(
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Article:\n{text[:4000]}"},
        ],
        max_tokens=220,
        temperature=0.3,
    )


def _local_demo_summary(article_text: str) -> str:
    first_sentence = article_text.replace("\n", " ").split(".")[0].strip()
    if not first_sentence:
        first_sentence = article_text[:140].strip()
    return (
        f"{first_sentence}. This is a demo summary because HF_TOKEN is not set. "
        "What this means for you: add Hugging Face credentials to hear live news."
    )
