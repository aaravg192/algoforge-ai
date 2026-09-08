from pathlib import Path

import numpy as np

from data_loader import load_problems
from document_builder import build_documents, save_documents
from embedding_service import EmbeddingService
from vector_store import VectorStore


BASE_PATH = Path(__file__).resolve().parent.parent

CLEANED_DATA_PATH = (
    BASE_PATH
    / "data"
    / "cleaned_problems.json"
)

EMBEDDINGS_PATH = (
    BASE_PATH
    / "data"
    / "problem_embeddings.npy"
)

FAISS_INDEX_PATH = (
    BASE_PATH
    / "data"
    / "problems.index"
)


def main():
    print("=== AlgoForge RAG Prototype ===")

    # --------------------------------------------------
    # 1. Load problems
    # --------------------------------------------------

    df = load_problems()

    # --------------------------------------------------
    # 2. Build cleaned RAG documents
    # --------------------------------------------------

    documents = build_documents(df)

    print(f"\nBuilt documents: {len(documents)}")

    # --------------------------------------------------
    # 3. Save cleaned documents
    # --------------------------------------------------

    save_documents(
        documents,
        CLEANED_DATA_PATH,
    )

    # --------------------------------------------------
    # 4. Load saved embeddings
    # --------------------------------------------------

    embeddings = np.load(EMBEDDINGS_PATH)

    print(
        f"\nLoaded embeddings: {embeddings.shape}"
    )

    # --------------------------------------------------
    # 5. Create FAISS vector store
    # --------------------------------------------------

    dimension = embeddings.shape[1]

    vector_store = VectorStore(dimension)

    # --------------------------------------------------
    # 6. Add embeddings
    # --------------------------------------------------

    vector_store.add(embeddings)

    print(
        f"FAISS index size: {vector_store.size}"
    )

    # --------------------------------------------------
    # 7. Save FAISS index
    # --------------------------------------------------

    vector_store.save(
        FAISS_INDEX_PATH
    )

    # --------------------------------------------------
    # 8. Test semantic search
    # --------------------------------------------------

    embedding_service = EmbeddingService()

    query = "problems involving hash maps"

    query_embedding = embedding_service.encode(
        [query]
    )

    scores, indices = vector_store.search(
        query_embedding,
        top_k=5,
    )

    print("\n=== SEARCH RESULTS ===")
    print(f"Query: {query}")

    for rank, (score, index) in enumerate(
        zip(scores[0], indices[0]),
        start=1,
    ):
        document = documents[index]

        print(
            f"\n{rank}. "
            f"{document['metadata']['title']}"
        )

        print(
            f"   Difficulty: "
            f"{document['metadata']['difficulty']}"
        )

        print(
            f"   Topics: "
            f"{', '.join(document['metadata']['topics'])}"
        )

        print(
            f"   Score: {score:.4f}"
        )


if __name__ == "__main__":
    main()