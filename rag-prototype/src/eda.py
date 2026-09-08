from pathlib import Path
import json
import ast

import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_PATH = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_PATH / "data" / "filtered_problems.json"
GRAPHS_PATH = BASE_PATH / "graphs"

GRAPHS_PATH.mkdir(parents=True, exist_ok=True)


# ============================================================
# LOAD DATA
# ============================================================

def load_data() -> list[dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Could not find filtered dataset:\n{DATA_PATH}"
        )

    with open(DATA_PATH, "r", encoding="utf-8") as file:
        documents = json.load(file)

    print(f"Loaded {len(documents)} problems")

    return documents


# ============================================================
# CONVERT JSON -> DATAFRAME
# ============================================================

def build_dataframe(documents: list[dict]) -> pd.DataFrame:
    rows = []

    for document in documents:
        text = document.get("text", {})
        metadata = document.get("metadata", {})

        rows.append({
            # -------------------------
            # Text fields
            # -------------------------
            "problem": text.get("problem"),
            "difficulty": text.get("difficulty"),
            "topics": text.get("topics", []),
            "description": text.get("description"),
            "examples": text.get("examples", []),
            "constraints": text.get("constraints", []),
            "follow_up": text.get("follow_up"),
            "hints": text.get("hints", []),
            "solution_explanation": text.get("solution_explanation"),
            "approaches": text.get("approaches", []),

            # -------------------------
            # Metadata
            # -------------------------
            "acceptance_rate": metadata.get("acceptance_rate"),
            "likes": metadata.get("likes"),
            "dislikes": metadata.get("dislikes"),
            "difficulty_meta": metadata.get("difficulty"),
            "category": metadata.get("category"),
            "paid_only": metadata.get("paid_only"),
            "title_slug": metadata.get("title_slug"),
        })

    df = pd.DataFrame(rows)

    # Prefer difficulty from text, fallback to metadata
    df["difficulty"] = df["difficulty"].fillna(df["difficulty_meta"])

    # Convert numeric columns
    df["acceptance_rate"] = pd.to_numeric(
        df["acceptance_rate"],
        errors="coerce"
    )

    df["likes"] = pd.to_numeric(
        df["likes"],
        errors="coerce"
    )

    df["dislikes"] = pd.to_numeric(
        df["dislikes"],
        errors="coerce"
    )

    return df


# ============================================================
# TOPIC PARSER
# ============================================================

def parse_topics(value) -> list[str]:
    """
    Handles topics stored as:
        ["Array", "Hash Table"]

    or:
        "['Array', 'Hash Table']"
    """

    if isinstance(value, list):
        return [str(topic).strip() for topic in value]

    if pd.isna(value):
        return []

    try:
        parsed = ast.literal_eval(str(value))

        if isinstance(parsed, list):
            return [str(topic).strip() for topic in parsed]

    except (ValueError, SyntaxError):
        pass

    return []


# ============================================================
# GRAPH 1
# PROBLEMS BY DIFFICULTY
# ============================================================

def plot_difficulty_distribution(df: pd.DataFrame):
    counts = df["difficulty"].value_counts()

    plt.figure(figsize=(8, 5))

    counts.plot(kind="bar")

    plt.title("Problems by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Number of Problems")

    plt.xticks(rotation=0)

    plt.tight_layout()

    path = GRAPHS_PATH / "difficulty_distribution.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 2
# TOP 20 TOPICS
# ============================================================

def plot_topic_distribution(df: pd.DataFrame):
    topic_counts = {}

    for topics in df["topics"]:
        topics = parse_topics(topics)

        for topic in topics:
            topic_counts[topic] = topic_counts.get(topic, 0) + 1

    topic_series = (
        pd.Series(topic_counts)
        .sort_values(ascending=False)
        .head(20)
        .sort_values()
    )

    plt.figure(figsize=(10, 8))

    topic_series.plot(kind="barh")

    plt.title("Top 20 Most Common Topics")
    plt.xlabel("Number of Problems")
    plt.ylabel("Topic")

    plt.tight_layout()

    path = GRAPHS_PATH / "topic_distribution.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 3
# TOPIC × DIFFICULTY HEATMAP
# ============================================================

def plot_topic_difficulty_heatmap(df: pd.DataFrame):
    rows = []

    for _, row in df.iterrows():

        topics = parse_topics(row["topics"])

        for topic in topics:
            rows.append({
                "topic": topic,
                "difficulty": row["difficulty"]
            })

    topic_df = pd.DataFrame(rows)

    if topic_df.empty:
        print("Skipping topic × difficulty heatmap: no topic data")
        return

    # Select top 15 topics
    top_topics = (
        topic_df["topic"]
        .value_counts()
        .head(15)
        .index
    )

    topic_df = topic_df[
        topic_df["topic"].isin(top_topics)
    ]

    pivot = pd.crosstab(
        topic_df["topic"],
        topic_df["difficulty"]
    )

    # Maintain useful difficulty ordering
    difficulty_order = [
        "Easy",
        "Medium",
        "Hard"
    ]

    existing_columns = [
        difficulty
        for difficulty in difficulty_order
        if difficulty in pivot.columns
    ]

    pivot = pivot[existing_columns]

    plt.figure(figsize=(10, 8))

    plt.imshow(
        pivot.values,
        aspect="auto"
    )

    plt.colorbar(label="Number of Problems")

    plt.xticks(
        range(len(pivot.columns)),
        pivot.columns
    )

    plt.yticks(
        range(len(pivot.index)),
        pivot.index
    )

    plt.title("Topic × Difficulty")

    plt.xlabel("Difficulty")
    plt.ylabel("Topic")

    # Add values inside cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            value = pivot.iloc[i, j]

            plt.text(
                j,
                i,
                str(value),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    path = GRAPHS_PATH / "topic_difficulty_heatmap.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 4
# ACCEPTANCE RATE DISTRIBUTION
# ============================================================

def plot_acceptance_distribution(df: pd.DataFrame):
    data = df["acceptance_rate"].dropna()

    if data.empty:
        print("Skipping acceptance distribution: no data")
        return

    plt.figure(figsize=(8, 5))

    plt.hist(
        data,
        bins=30
    )

    plt.title("Acceptance Rate Distribution")
    plt.xlabel("Acceptance Rate")
    plt.ylabel("Number of Problems")

    plt.tight_layout()

    path = GRAPHS_PATH / "acceptance_distribution.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 5
# ACCEPTANCE RATE BY DIFFICULTY
# ============================================================

def plot_acceptance_by_difficulty(df: pd.DataFrame):
    data = df[
        ["difficulty", "acceptance_rate"]
    ].dropna()

    if data.empty:
        print("Skipping acceptance by difficulty: no data")
        return

    difficulties = [
        "Easy",
        "Medium",
        "Hard"
    ]

    grouped_data = []

    labels = []

    for difficulty in difficulties:

        values = data[
            data["difficulty"] == difficulty
        ]["acceptance_rate"].dropna()

        if not values.empty:
            grouped_data.append(values)
            labels.append(difficulty)

    if not grouped_data:
        return

    plt.figure(figsize=(8, 5))

    plt.boxplot(
        grouped_data,
        labels=labels
    )

    plt.title("Acceptance Rate by Difficulty")
    plt.xlabel("Difficulty")
    plt.ylabel("Acceptance Rate")

    plt.tight_layout()

    path = GRAPHS_PATH / "acceptance_by_difficulty.png"

    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 6
# LIKES VS DISLIKES
# ============================================================

def plot_likes_vs_dislikes(df: pd.DataFrame):
    data = df[
        ["likes", "dislikes"]
    ].dropna()

    if data.empty:
        print("Skipping likes vs dislikes: no data")
        return

    plt.figure(figsize=(8, 6))

    plt.scatter(
        data["likes"],
        data["dislikes"],
        alpha=0.5
    )

    plt.title("Likes vs Dislikes")
    plt.xlabel("Likes")
    plt.ylabel("Dislikes")

    plt.tight_layout()

    path = GRAPHS_PATH / "likes_vs_dislikes.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 7
# PROBLEMS BY CATEGORY
# ============================================================

def plot_category_distribution(df: pd.DataFrame):
    counts = df["category"].value_counts()

    if counts.empty:
        print("Skipping category distribution: no data")
        return

    plt.figure(figsize=(10, 6))

    counts.plot(kind="bar")

    plt.title("Problems by Category")
    plt.xlabel("Category")
    plt.ylabel("Number of Problems")

    plt.xticks(rotation=45, ha="right")

    plt.tight_layout()

    path = GRAPHS_PATH / "category_distribution.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# GRAPH 8
# MISSING DATA
# ============================================================

def plot_missing_values(df: pd.DataFrame):
    # Count missing values
    missing_count = df.isnull().sum()

    # Calculate missing percentage
    missing_percentage = (
        missing_count / len(df) * 100
    )

    # Build a DataFrame for easier analysis
    missing = pd.DataFrame({
        "missing_count": missing_count,
        "missing_percentage": missing_percentage
    })

    # Keep only columns that actually have missing values
    missing = missing[
        missing["missing_count"] > 0
    ].sort_values(
        "missing_count",
        ascending=False
    )

    print("\nMissing Data:")
    print(missing)

    # Nothing missing
    if missing.empty:
        print("No missing values found")
        return

    # Plot missing counts
    plt.figure(figsize=(10, 7))

    missing["missing_count"].sort_values().plot(
        kind="barh"
    )

    plt.title("Missing Values by Field")
    plt.xlabel("Number of Missing Values")
    plt.ylabel("Field")

    plt.tight_layout()

    path = GRAPHS_PATH / "missing_values.png"

    plt.savefig(path, dpi=300)

    plt.close()

    print(f"Saved: {path}")


# ============================================================
# DATASET SUMMARY
# ============================================================

def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)

    print(f"Total Problems       : {len(df)}")

    print("\nDifficulty:")
    print(df["difficulty"].value_counts())

    print("\nCategories:")
    print(df["category"].value_counts())

    print("\nMissing Values:")
    print(
        df.isnull()
        .sum()
        .sort_values(ascending=False)
    )

    print("\nAcceptance Rate:")
    print(df["acceptance_rate"].describe())


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 60)
    print("AlgoForge RAG - Exploratory Data Analysis")
    print("=" * 60)

    # Load
    documents = load_data()

    # Convert to DataFrame
    df = build_dataframe(documents)

    print_summary(df)

    print("\n" + "=" * 60)
    print("GENERATING GRAPHS")
    print("=" * 60)

    # 1
    plot_difficulty_distribution(df)

    # 2
    plot_topic_distribution(df)

    # 3
    plot_topic_difficulty_heatmap(df)

    # 4
    plot_acceptance_distribution(df)

    # 5
    plot_acceptance_by_difficulty(df)

    # 6
    plot_likes_vs_dislikes(df)

    # 7
    plot_category_distribution(df)

    # 8
    plot_missing_values(df)

    print("\n" + "=" * 60)
    print("EDA COMPLETE")
    print("=" * 60)

    print(f"\nGraphs saved to:")
    print(GRAPHS_PATH)


if __name__ == "__main__":
    main()