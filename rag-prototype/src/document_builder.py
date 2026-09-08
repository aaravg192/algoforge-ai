import ast
import json
import re
from typing import Any

import pandas as pd


def clean_html(text: Any) -> str:
    """Convert HTML content into clean readable text."""

    if pd.isna(text):
        return ""

    text = str(text)

    # Remove video/iframe content completely
    text = re.sub(
        r"<iframe.*?</iframe>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Preserve superscript/subscript meaning
    text = re.sub(
        r"<sup>(.*?)</sup>",
        r"^\1",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    text = re.sub(
        r"<sub>(.*?)</sub>",
        r"_\1",
        text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    # Preserve common block boundaries
    text = re.sub(
        r"<br\s*/?>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"</p>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    text = re.sub(
        r"</li>",
        "\n",
        text,
        flags=re.IGNORECASE,
    )

    # Remove remaining HTML tags
    text = re.sub(r"<[^>]+>", " ", text)

    # Decode common HTML entities
    replacements = {
        "&nbsp;": " ",
        "&lt;": "<",
        "&gt;": ">",
        "&amp;": "&",
        "&quot;": '"',
        "&#39;": "'",
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    # Remove dataset-specific noise
    text = re.sub(r"\[TOC\]", "", text, flags=re.IGNORECASE)
    text = re.sub(
        r"##\s*Video Solution",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(
        r"##\s*Solution Article",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Normalize whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    return text.strip()


def parse_list(value: Any) -> list:
    """Parse stringified Python/JSON lists safely."""

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return []


def build_problem_document(row: pd.Series) -> dict:
    """Convert one dataframe row into a RAG-ready document."""

    topics = parse_list(row["topics"])
    hints = parse_list(row["hints"])
    similar_questions = parse_list(row["similar_questions"])

    topic_names = [
        str(topic)
        for topic in topics
    ]

    hint_text = [
        clean_html(hint)
        for hint in hints
        if hint
    ]

    similar_titles = [
        str(question.get("title", ""))
        for question in similar_questions
        if isinstance(question, dict)
    ]

    description = clean_html(row["description"])
    solution = clean_html(row["solution"])

    document_text = f"""
Problem: {row["title"]}

Difficulty: {row["difficulty"]}

Topics: {", ".join(topic_names)}

Description:
{description}

Hints:
{"\n".join(f"- {hint}" for hint in hint_text)}

Solution Explanation:
{solution}

Similar Problems:
{", ".join(similar_titles)}
""".strip()

    metadata = {
        "frontend_question_id": str(row["frontendQuestionId"]),
        "title": row["title"],
        "title_slug": row["titleSlug"],
        "difficulty": row["difficulty"],
        "paid_only": bool(row["paidOnly"]),
        "category": row["category"],
        "topics": topic_names,
        "acceptance_rate": row["acceptance_rate"],
        "likes": row["likes"],
        "dislikes": row["dislikes"],
        "url": row["url"],
    }

    return {
        "text": document_text,
        "metadata": metadata,
    }


def build_documents(df: pd.DataFrame) -> list[dict]:
    """Convert the complete dataframe into RAG documents."""

    documents = []

    for _, row in df.iterrows():
        document = build_problem_document(row)
        documents.append(document)

    return documents

def save_documents(
    documents: list[dict],
    path: str,
) -> None:
    """Save cleaned RAG documents to a JSON file."""

    with open(path, "w", encoding="utf-8") as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(f"Saved cleaned documents to: {path}")