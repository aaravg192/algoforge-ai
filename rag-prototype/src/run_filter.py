from pathlib import Path

from filter_pipeline import (
    load_cleaned_documents,
    filter_all_documents,
    save_filtered_documents,
)


BASE_PATH = (
    Path(__file__).resolve().parent.parent
)

INPUT_PATH = (
    BASE_PATH
    / "data"
    / "cleaned_problems.json"
)

OUTPUT_PATH = (
    BASE_PATH
    / "data"
    / "filtered_problems.json"
)


def main():
    print("=== AlgoForge Data Filtering Pipeline ===")

    # 1. Load cleaned documents
    documents = load_cleaned_documents(
        INPUT_PATH
    )

    print(
        f"Loaded documents: {len(documents)}"
    )

    # 2. Filter and structure
    filtered_documents = filter_all_documents(
        documents
    )

    print(
        f"Filtered documents: "
        f"{len(filtered_documents)}"
    )

    # 3. Save
    save_filtered_documents(
        filtered_documents,
        OUTPUT_PATH,
    )

    print("\n=== Pipeline Complete ===")


if __name__ == "__main__":
    main()