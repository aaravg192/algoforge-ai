from pathlib import Path

from context_builder import ContextBuilder
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
        "=== AlgoForge Context Builder Test ==="
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
    # Load FAISS vector store
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
    # Create context builder
    # --------------------------------------------------------

    context_builder = ContextBuilder()

    # ========================================================
    # Recommendation Context Test
    # ========================================================

    query = (
        "I want an easy problem "
        "involving hash maps"
    )

    results = retriever.retrieve(
        query,
        top_k=3,
        difficulty="Easy",
        topics=["Hash Table"],
        paid_only=False,
    )

    recommendation_context = (
        context_builder.build_recommendation_context(
            results
        )
    )

    print(
        "\n\n========================================================"
    )

    print(
        "RECOMMENDATION CONTEXT TEST"
    )

    print(
        "========================================================"
    )

    print(
        f"\nQuery: {query}"
    )

    print(
        "\n--- GENERATED CONTEXT ---\n"
    )

    print(
        recommendation_context
    )


if __name__ == "__main__":
    main()