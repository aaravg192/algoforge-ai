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


def display_results(
    results,
    test_name,
    query,
):
    """
    Display retrieved problems in a consistent format.
    """

    print(
        f"\n=== {test_name} ==="
    )

    print(
        f"Query: {query}"
    )

    if not results:
        print(
            "\nNo results found."
        )
        return

    for rank, result in enumerate(
        results,
        start=1,
    ):

        document = result["document"]

        text = document["text"]

        metadata = document["metadata"]

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
            f"   Paid: "
            f"{metadata['paid_only']}"
        )

        print(
            f"   Similarity: "
            f"{result['score']:.4f}"
        )

        print(
            f"   ID: "
            f"{metadata.get('frontend_question_id')}"
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

    # ========================================================
    # TEST 1
    # Pure semantic retrieval
    # ========================================================

    query = (
        "I want problems involving "
        "hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
    )

    display_results(
        results=results,
        test_name="TEST 1: SEMANTIC RETRIEVAL",
        query=query,
    )

    # ========================================================
    # TEST 2
    # Semantic retrieval + difficulty filter
    # ========================================================

    query = (
        "I want problems involving "
        "hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
        difficulty="Easy",
    )

    display_results(
        results=results,
        test_name="TEST 2: EASY HASH MAP PROBLEMS",
        query=query,
    )

    # ========================================================
    # TEST 3
    # Semantic retrieval + topic filter
    # ========================================================

    query = (
        "I want problems involving "
        "hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
        topics=["Hash Table"],
    )

    display_results(
        results=results,
        test_name="TEST 3: HASH TABLE PROBLEMS",
        query=query,
    )

    # ========================================================
    # TEST 4
    # Semantic retrieval + multiple filters
    # ========================================================

    query = (
        "I want problems involving "
        "hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=5,
        difficulty="Easy",
        topics=["Hash Table"],
        paid_only=False,
    )

    display_results(
        results=results,
        test_name="TEST 4: HYBRID RETRIEVAL",
        query=query,
    )


if __name__ == "__main__":
    main()