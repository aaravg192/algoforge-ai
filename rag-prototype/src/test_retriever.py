from pathlib import Path

from embedding_service import EmbeddingService
from filter_pipeline import load_cleaned_documents
from retriever import ProblemRetriever
from vector_store import VectorStore


BASE_PATH = (
    Path(__file__).resolve().parent.parent
)

DOCUMENTS_PATH = (
    BASE_PATH
    / "data"
    / "filtered_problems.json"
)

INDEX_PATH = (
    BASE_PATH
    / "data"
    / "problems.index"
)


def main():

    print(
        "=== AlgoForge RAG Retrieval Test ==="
    )

    # --------------------------------------------------------
    # Load documents
    # --------------------------------------------------------

    documents = load_cleaned_documents(
        DOCUMENTS_PATH
    )

    print(
        f"Loaded documents: {len(documents)}"
    )

    # --------------------------------------------------------
    # Load vector store
    # --------------------------------------------------------

    vector_store = VectorStore.load(
        INDEX_PATH
    )

    print(
        f"Loaded FAISS index: "
        f"{vector_store.size} vectors"
    )

    # --------------------------------------------------------
    # Load embedding model
    # --------------------------------------------------------

    embedding_service = (
        EmbeddingService()
    )

    # --------------------------------------------------------
    # Create retriever
    # --------------------------------------------------------

    retriever = ProblemRetriever(
        embedding_service=embedding_service,
        vector_store=vector_store,
        documents=documents,
    )

    # --------------------------------------------------------
    # Test query
    # --------------------------------------------------------

    query = (
        "I want problems involving "
        "hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print(
        "\n=== RETRIEVAL RESULTS ==="
    )

    print(
        f"Query: {query}"
    )

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result[
            "document"
        ]

        text = document[
            "text"
        ]

        metadata = document[
            "metadata"
        ]

        print(
            f"\n{rank}. "
            f"{text['problem']}"
        )

        print(
            f"   Difficulty: "
            f"{text['difficulty']}"
        )

        print(
            f"   Topics: "
            f"{', '.join(text['topics'])}"
        )

        print(
            f"   Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"   ID: "
            f"{metadata.get('frontend_question_id')}"
        )


if __name__ == "__main__":
    main()