from typing import Any


class ContextBuilder:
    """
    Builds clean LLM-ready context for
    AlgoForge problem recommendations.
    """

    def build_recommendation_context(
        self,
        results: list[dict[str, Any]],
    ) -> str:
        """
        Build context for problem recommendation.

        Includes:
        - Problem title
        - Difficulty
        - Topics
        - Description
        - Constraints
        """

        if not results:
            return "No relevant problems were found."

        context_parts = []

        for rank, result in enumerate(
            results,
            start=1,
        ):

            document = result["document"]

            text = document["text"]

            problem = text["problem"]
            difficulty = text["difficulty"]
            topics = text["topics"]
            description = text["description"]
            constraints = text["constraints"]

            # ------------------------------------------------
            # Remove duplicate constraints
            # ------------------------------------------------

            unique_constraints = []

            for constraint in constraints:
                constraint = str(constraint).strip()

                if constraint and constraint not in unique_constraints:
                    unique_constraints.append(
                        constraint
                    )

            # ------------------------------------------------
            # Build problem context
            # ------------------------------------------------

            context_parts.append(
                f"""
Problem {rank}:

Title: {problem}

Difficulty: {difficulty}

Topics: {", ".join(topics)}

Description:
{description}

Constraints:
{"\n".join(f"- {constraint}" for constraint in unique_constraints)}
""".strip()
            )

        return "\n\n--------------------------------\n\n".join(
            context_parts
        )