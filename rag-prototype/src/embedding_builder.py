from typing import Any


def build_embedding_text(
    document: dict[str, Any],
) -> str:
    """
    Convert a filtered problem document into semantic text
    suitable for embedding.
    """

    text = document.get("text", {})

    problem = text.get(
        "problem",
        "",
    )

    difficulty = text.get(
        "difficulty",
        "",
    )

    topics = text.get(
        "topics",
        [],
    )

    description = text.get(
        "description",
        "",
    )

    examples = text.get(
        "examples",
        [],
    )

    constraints = text.get(
        "constraints",
        [],
    )

    follow_up = text.get(
        "follow_up",
        None,
    )

    hints = text.get(
        "hints",
        [],
    )

    solution_explanation = text.get(
        "solution_explanation",
        "",
    )

    approaches = text.get(
        "approaches",
        [],
    )

    parts = []

    # --------------------------------------------------------
    # Problem identity
    # --------------------------------------------------------

    if problem:
        parts.append(
            f"Problem: {problem}"
        )

    if difficulty:
        parts.append(
            f"Difficulty: {difficulty}"
        )

    if topics:
        parts.append(
            f"Topics: {', '.join(topics)}"
        )

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    if description:
        parts.append(
            f"Description:\n{description}"
        )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    if examples:

        example_text = []

        for index, example in enumerate(
            examples,
            start=1,
        ):
            example_text.append(
                f"""
Example {index}:
Input: {example.get("input", "")}
Output: {example.get("output", "")}
Explanation: {example.get("explanation", "")}
""".strip()
            )

        parts.append(
            "Examples:\n"
            + "\n\n".join(example_text)
        )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    if constraints:

        constraint_text = "\n".join(
            f"- {constraint}"
            for constraint in constraints
        )

        parts.append(
            f"Constraints:\n{constraint_text}"
        )

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    if follow_up:
        parts.append(
            f"Follow-up:\n{follow_up}"
        )

    # --------------------------------------------------------
    # Hints
    # --------------------------------------------------------

    if hints:

        hint_text = "\n".join(
            f"- {hint}"
            for hint in hints
        )

        parts.append(
            f"Hints:\n{hint_text}"
        )

    # --------------------------------------------------------
    # Solution explanation
    # --------------------------------------------------------

    if solution_explanation:
        parts.append(
            f"Solution Explanation:\n"
            f"{solution_explanation}"
        )

    # --------------------------------------------------------
    # Approaches
    # --------------------------------------------------------

    if approaches:

        approach_text = []

        for approach in approaches:

            name = approach.get(
                "name",
                "",
            )

            intuition = approach.get(
                "intuition",
                "",
            )

            algorithm = approach.get(
                "algorithm",
                "",
            )

            implementation = approach.get(
                "implementation",
                "",
            )

            complexity = approach.get(
                "complexity",
                {},
            )

            time = complexity.get(
                "time",
                "",
            )

            space = complexity.get(
                "space",
                "",
            )

            approach_text.append(
                f"""
Approach: {name}

Intuition:
{intuition}

Algorithm:
{algorithm}

Implementation:
{implementation}

Complexity:
Time: {time}
Space: {space}
""".strip()
            )

        parts.append(
            "Approaches:\n"
            + "\n\n".join(approach_text)
        )

    return "\n\n".join(parts)