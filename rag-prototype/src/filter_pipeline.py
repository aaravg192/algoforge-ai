import json
import re
from pathlib import Path
from typing import Any


# ============================================================
# BASIC HELPERS
# ============================================================

def clean_text(text: Any) -> str:
    """
    Normalize whitespace while preserving meaningful line breaks.
    """

    if text is None:
        return ""

    text = str(text)

    if not text.strip():
        return ""

    text = text.replace("\r\n", "\n")
    text = text.replace("\r", "\n")

    # Normalize spaces/tabs without destroying newlines
    text = re.sub(r"[ \t]+", " ", text)

    # Remove spaces around newlines
    text = re.sub(r" *\n *", "\n", text)

    # Remove excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def remove_markdown_noise(text: str) -> str:
    """
    Remove markdown elements that are structural noise
    while preserving useful content.
    """

    if not text:
        return ""

    # Horizontal rules
    text = re.sub(
        r"^\s*---+\s*$",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Markdown headings
    text = re.sub(
        r"^\s*#{1,6}\s+",
        "",
        text,
        flags=re.MULTILINE,
    )

    # Bold section markers
    text = re.sub(
        r"\*\*(.*?)\*\*",
        r"\1",
        text,
    )

    # Inline code markers
    text = re.sub(
        r"`([^`]+)`",
        r"\1",
        text,
    )

    return clean_text(text)


def clean_list_item(text: str) -> str:
    """
    Remove common markdown list prefixes.
    """

    text = text.strip()

    text = re.sub(
        r"^[-*+]\s+",
        "",
        text,
    )

    text = re.sub(
        r"^\d+[.)]\s+",
        "",
        text,
    )

    return text.strip()


# ============================================================
# SECTION EXTRACTION
# ============================================================

def extract_section(
    text: str,
    start_pattern: str,
    end_patterns: list[str],
) -> str:
    """
    Extract text between a starting section heading and the
    next known section heading.
    """

    if not text:
        return ""

    start_match = re.search(
        start_pattern,
        text,
        flags=re.IGNORECASE | re.MULTILINE,
    )

    if not start_match:
        return ""

    start = start_match.end()

    remaining = text[start:]

    end_positions = []

    for pattern in end_patterns:
        match = re.search(
            pattern,
            remaining,
            flags=re.IGNORECASE | re.MULTILINE,
        )

        if match:
            end_positions.append(match.start())

    if end_positions:
        end = min(end_positions)
        section = remaining[:end]
    else:
        section = remaining

    return remove_markdown_noise(section)


# ============================================================
# EXAMPLES
# ============================================================

def extract_examples(
    description: str,
) -> list[dict[str, str]]:
    """
    Extract Example blocks from a problem description.

    Expected format:

    Example 1:

    Input: ...
    Output: ...
    Explanation: ...

    Example 2:

    Input: ...
    Output: ...
    Explanation: ...
    """

    if not description:
        return []

    examples = []

    pattern = re.compile(
        r"Example\s+\d+\s*:\s*(.*?)(?="
        r"Example\s+\d+\s*:|"
        r"Constraints\s*:|"
        r"Follow-up\s*:|"
        r"\Z"
        r")",
        flags=re.IGNORECASE | re.DOTALL,
    )

    matches = pattern.findall(description)

    for block in matches:
        block = clean_text(block)

        input_match = re.search(
            r"Input\s*:\s*(.*?)(?=\s*Output\s*:|\Z)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        output_match = re.search(
            r"Output\s*:\s*(.*?)(?=\s*Explanation\s*:|\Z)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        explanation_match = re.search(
            r"Explanation\s*:\s*(.*)",
            block,
            flags=re.IGNORECASE | re.DOTALL,
        )

        example = {
            "input": clean_text(
                input_match.group(1)
                if input_match
                else ""
            ),
            "output": clean_text(
                output_match.group(1)
                if output_match
                else ""
            ),
            "explanation": clean_text(
                explanation_match.group(1)
                if explanation_match
                else ""
            ),
        }

        # Only keep an example if something meaningful
        # was actually extracted.
        if any(example.values()):
            examples.append(example)

    return examples


# ============================================================
# CONSTRAINTS
# ============================================================

def extract_constraints(
    description: str,
) -> list[str]:
    """
    Extract individual constraints from the description.
    """

    if not description:
        return []

    section = extract_section(
        description,
        r"Constraints\s*:",
        [
            r"Follow-up\s*:",
        ],
    )

    if not section:
        return []

    constraints = []

    # Constraints are normally separated by newlines.
    for line in section.split("\n"):
        line = clean_list_item(line)

        if not line:
            continue

        constraints.append(line)

    return constraints


# ============================================================
# FOLLOW-UP
# ============================================================

def extract_follow_up(
    description: str,
) -> str | None:
    """
    Extract the optional follow-up question.
    """

    if not description:
        return None

    match = re.search(
        r"Follow-up\s*:\s*(.*)",
        description,
        flags=re.IGNORECASE | re.DOTALL,
    )

    if not match:
        return None

    follow_up = clean_text(
        match.group(1)
    )

    return follow_up if follow_up else None


# ============================================================
# HINTS
# ============================================================

def extract_hints(
    text: str,
) -> list[str]:
    """
    Extract hints from the Hints section.
    """

    if not text:
        return []

    section = extract_section(
        text,
        r"Hints\s*:",
        [
            r"Solution Explanation\s*:",
            r"Similar Problems\s*:",
        ],
    )

    if not section:
        return []

    hints = []

    for line in section.split("\n"):
        line = clean_list_item(line)

        if line:
            hints.append(line)

    return hints


# ============================================================
# SUBSECTION EXTRACTION
# ============================================================

def extract_subsection(
    text: str,
    section_name: str,
    end_sections: list[str],
) -> str:
    """
    Extract a markdown subsection such as:

    **Algorithm**

    ...

    **Implementation**

    ...
    """

    if not text:
        return ""

    pattern = (
        rf"\*\*\s*{re.escape(section_name)}\s*\*\*"
    )

    match = re.search(
        pattern,
        text,
        flags=re.IGNORECASE,
    )

    if not match:
        return ""

    start = match.end()

    remaining = text[start:]

    end_positions = []

    for section in end_sections:
        end_match = re.search(
            rf"\*\*\s*{re.escape(section)}\s*\*\*",
            remaining,
            flags=re.IGNORECASE,
        )

        if end_match:
            end_positions.append(
                end_match.start()
            )

    if end_positions:
        end = min(end_positions)
        content = remaining[:end]
    else:
        content = remaining

    return remove_markdown_noise(content)


# ============================================================
# COMPLEXITY
# ============================================================

def extract_complexity(
    text: str,
) -> dict[str, str]:
    """
    Extract time and space complexity from an approach.
    """

    time = ""
    space = ""

    if not text:
        return {
            "time": time,
            "space": space,
        }

    # Supports:
    # Time complexity: O(n)
    # Time Complexity: O(n)
    # Time: O(n)

    time_match = re.search(
        r"(?:Time\s+complexity|Time)\s*:\s*(.*?)(?=\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    space_match = re.search(
        r"(?:Space\s+complexity|Space)\s*:\s*(.*?)(?=\n|$)",
        text,
        flags=re.IGNORECASE,
    )

    if time_match:
        time = clean_text(
            time_match.group(1)
        )

    if space_match:
        space = clean_text(
            space_match.group(1)
        )

    return {
        "time": time,
        "space": space,
    }


# ============================================================
# APPROACH EXTRACTION
# ============================================================

def extract_approaches(
    solution_text: str,
) -> list[dict[str, Any]]:
    """
    Extract solution approaches and their subsections.

    Expected source format:

    ### Approach 1: Brute Force

    **Intuition**

    ...

    **Algorithm**

    ...

    **Implementation**

    ...

    **Complexity Analysis**

    Time complexity: O(n²)
    Space complexity: O(1)

    ### Approach 2: Hash Map

    ...
    """

    if not solution_text:
        return []

    approach_pattern = re.compile(
        r"###\s*Approach\s+\d+\s*:\s*(.*?)(?="
        r"###\s*Approach\s+\d+\s*:|"
        r"\Z"
        r")",
        flags=re.IGNORECASE | re.DOTALL,
    )

    approaches = []

    for match in approach_pattern.finditer(
        solution_text
    ):
        block = match.group(0)

        # ----------------------------------------------------
        # Approach name
        # ----------------------------------------------------

        name_match = re.search(
            r"###\s*Approach\s+\d+\s*:\s*(.*)",
            block,
            flags=re.IGNORECASE,
        )

        name = (
            clean_text(
                name_match.group(1)
            )
            if name_match
            else ""
        )

        # ----------------------------------------------------
        # Subsections
        # ----------------------------------------------------

        intuition = extract_subsection(
            block,
            "Intuition",
            [
                "Algorithm",
                "Implementation",
                "Complexity Analysis",
            ],
        )

        algorithm = extract_subsection(
            block,
            "Algorithm",
            [
                "Intuition",
                "Implementation",
                "Complexity Analysis",
            ],
        )

        implementation = extract_subsection(
            block,
            "Implementation",
            [
                "Intuition",
                "Algorithm",
                "Complexity Analysis",
            ],
        )

        complexity_section = extract_subsection(
            block,
            "Complexity Analysis",
            [],
        )

        complexity = extract_complexity(
            complexity_section
        )

        approaches.append(
            {
                "name": name,
                "intuition": intuition,
                "algorithm": algorithm,
                "implementation": implementation,
                "complexity": complexity,
            }
        )

    return approaches


# ============================================================
# SOLUTION SECTION
# ============================================================

def extract_solution(
    raw_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    """
    Extract the general solution explanation and all
    approaches from the Solution Explanation section.

    This keeps the two pieces of information separate:

    solution_explanation
        General explanation before Approach 1.

    approaches
        Structured approach objects.
    """

    solution_section = extract_section(
        raw_text,
        r"Solution Explanation\s*:",
        [
            r"Similar Problems\s*:",
        ],
    )

    if not solution_section:
        return "", []

    # Find the first approach.
    first_approach_match = re.search(
        r"###\s*Approach\s+\d+\s*:",
        solution_section,
        flags=re.IGNORECASE,
    )

    if first_approach_match:
        general_explanation = solution_section[
            :first_approach_match.start()
        ]

        approaches_text = solution_section[
            first_approach_match.start():
        ]
    else:
        general_explanation = solution_section
        approaches_text = ""

    general_explanation = remove_markdown_noise(
        general_explanation
    )

    approaches = extract_approaches(
        approaches_text
    )

    return (
        general_explanation,
        approaches,
    )


# ============================================================
# MAIN FILTER
# ============================================================

def filter_problem(
    document: dict,
) -> dict:
    """
    Convert one raw RAG document into structured ML-ready data.

    Output:

    {
        "text": {
            "problem": "...",
            "difficulty": "...",
            "topics": [...],
            "description": "...",
            "examples": [...],
            "constraints": [...],
            "follow_up": "...",
            "hints": [...],
            "solution_explanation": "...",
            "approaches": [...]
        },
        "metadata": {
            ...
        }
    }

    Metadata is preserved exactly from the source document.
    """

    raw_text = document.get(
        "text",
        "",
    )

    metadata = document.get(
        "metadata",
        {},
    )

    # --------------------------------------------------------
    # Problem
    # --------------------------------------------------------

    problem = extract_section(
        raw_text,
        r"Problem\s*:",
        [
            r"Difficulty\s*:",
        ],
    )

    # --------------------------------------------------------
    # Difficulty
    # --------------------------------------------------------

    difficulty = extract_section(
        raw_text,
        r"Difficulty\s*:",
        [
            r"Topics\s*:",
        ],
    )

    # --------------------------------------------------------
    # Topics
    # --------------------------------------------------------

    topics_text = extract_section(
        raw_text,
        r"Topics\s*:",
        [
            r"Description\s*:",
        ],
    )

    topics = [
        topic.strip()
        for topic in topics_text.split(",")
        if topic.strip()
    ]

    # --------------------------------------------------------
    # Description
    # --------------------------------------------------------

    description = extract_section(
        raw_text,
        r"Description\s*:",
        [
            r"Hints\s*:",
            r"Solution Explanation\s*:",
            r"Similar Problems\s*:",
        ],
    )

    # --------------------------------------------------------
    # Examples
    # --------------------------------------------------------

    examples = extract_examples(
        description
    )

    # --------------------------------------------------------
    # Constraints
    # --------------------------------------------------------

    constraints = extract_constraints(
        description
    )

    # --------------------------------------------------------
    # Follow-up
    # --------------------------------------------------------

    follow_up = extract_follow_up(
        description
    )

    # --------------------------------------------------------
    # Hints
    # --------------------------------------------------------

    hints = extract_hints(
        raw_text
    )

    # --------------------------------------------------------
    # Solution + Approaches
    # --------------------------------------------------------

    (
        solution_explanation,
        approaches,
    ) = extract_solution(
        raw_text
    )

    # --------------------------------------------------------
    # Final structured text
    # --------------------------------------------------------

    filtered_text = {
        "problem": problem,
        "difficulty": difficulty,
        "topics": topics,
        "description": description,
        "examples": examples,
        "constraints": constraints,
        "follow_up": follow_up,
        "hints": hints,
        "solution_explanation": solution_explanation,
        "approaches": approaches,
    }

    # --------------------------------------------------------
    # Preserve metadata exactly
    # --------------------------------------------------------

    return {
        "text": filtered_text,
        "metadata": metadata,
    }


# ============================================================
# FILE IO
# ============================================================

def load_cleaned_documents(
    path: str | Path,
) -> list[dict]:
    """
    Load cleaned_problems.json.
    """

    path = Path(path)

    with open(
        path,
        "r",
        encoding="utf-8",
    ) as file:
        return json.load(file)


def save_filtered_documents(
    documents: list[dict],
    path: str | Path,
) -> None:
    """
    Save filtered structured documents.
    """

    path = Path(path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        path,
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            documents,
            file,
            ensure_ascii=False,
            indent=2,
        )

    print(
        f"Saved filtered documents to: {path}"
    )


def filter_all_documents(
    documents: list[dict],
) -> list[dict]:
    """
    Filter the complete dataset.
    """

    filtered = []

    for document in documents:
        filtered.append(
            filter_problem(document)
        )

    return filtered