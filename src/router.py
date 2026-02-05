"""Query routing and orchestration for multi-domain knowledge base."""

import os
from typing import Literal, Optional
from anthropic import Anthropic

import sys
sys.path.insert(0, str(__file__).rsplit("/src", 1)[0])
from config import ANTHROPIC_API_KEY, DOMAINS

DomainType = Literal["seo", "web_builder", "all", "unknown"]

# Only print routing messages in CLI mode (not MCP/Slack)
_quiet_mode = os.environ.get("SEO_KB_QUIET", "0") == "1"


class QueryRouter:
    """Routes queries to the appropriate domain(s) based on content classification."""

    def __init__(self):
        """Initialize the query router."""
        self.anthropic = Anthropic(api_key=ANTHROPIC_API_KEY)
        self._domain_descriptions = "\n".join([
            f"- {name}: {config['description']}"
            for name, config in DOMAINS.items()
        ])

    def classify(self, question: str) -> DomainType:
        """
        Classify which domain(s) a question belongs to.
        Uses Claude Haiku for fast, cost-effective classification.

        Args:
            question: The user's question

        Returns:
            One of: 'seo', 'web_builder', 'all', or 'unknown'
        """
        classification_prompt = f"""Classify this question into exactly one category.

Available domains:
{self._domain_descriptions}

Categories:
- seo: Primarily about SEO, rankings, search engines, local SEO, backlinks
- web_builder: Primarily about website builders, web design, no-code tools, page builders
- all: Spans both domains or could benefit from knowledge in both
- unknown: Doesn't fit either domain well

Question: "{question}"

Respond with ONLY the category name (seo, web_builder, all, or unknown):"""

        try:
            response = self.anthropic.messages.create(
                model="claude-3-haiku-20240307",  # Fast and cheap for classification
                max_tokens=10,
                messages=[{"role": "user", "content": classification_prompt}]
            )

            result = response.content[0].text.strip().lower()

            # Validate response
            if result in ("seo", "web_builder", "all", "unknown"):
                return result
            return "unknown"
        except Exception:
            # On any error, default to querying all domains
            return "all"

    def route(
        self,
        question: str,
        explicit_domain: Optional[str] = None
    ) -> list[str]:
        """
        Determine which domains to query.

        Args:
            question: The user's question
            explicit_domain: If provided, overrides classification.
                           Use 'all' to query all domains.

        Returns:
            List of domain names to query
        """
        # Explicit domain takes precedence
        if explicit_domain:
            if explicit_domain == "all":
                return list(DOMAINS.keys())
            if explicit_domain in DOMAINS:
                return [explicit_domain]

        # Otherwise, classify the question
        classification = self.classify(question)

        if classification == "all" or classification == "unknown":
            return list(DOMAINS.keys())

        return [classification]


# Singleton instance
_router: QueryRouter | None = None


def get_router() -> QueryRouter:
    """Get the singleton query router instance."""
    global _router
    if _router is None:
        _router = QueryRouter()
    return _router
