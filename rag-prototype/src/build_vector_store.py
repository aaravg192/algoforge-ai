from pathlib import Path

import numpy as np

from vector_store import VectorStore


BASE_PATH = (
    Path(__file__).resolve().parent.parent
)

EMBEDDINGS_PATH = (
    BASE_PATH
    / "data"
    / "problem_embeddings.npy"
)

INDEX_PATH = (
    BASE_PATH
    / "data"
    / "problems.index"
)


def main():

    print(
        "=== AlgoForge Vector Store Builder ==="
    )

    # --------------------------------------------------------
    # Load embeddings
    # --------------------------------------------------------

    embeddings = np.load(
        EMBEDDINGS_PATH
    )

    print(
        f"Loaded embeddings: {embeddings.shape}"
    )

    # --------------------------------------------------------
    # Create FAISS store
    # --------------------------------------------------------

    dimension = embeddings.shape[1]

    vector_store = VectorStore(
        dimension
    )

    # --------------------------------------------------------
    # Add embeddings
    # --------------------------------------------------------

    vector_store.add(
        embeddings
    )

    print(
        f"Vectors in index: "
        f"{vector_store.size}"
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    vector_store.save(
        INDEX_PATH
    )

    print(
        "\n=== Vector Store Complete ==="
    )


if __name__ == "__main__":
    main()