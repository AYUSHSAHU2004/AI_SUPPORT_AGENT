"""
app/retrieval.py — ChromaDB vector store wrapper via LangChain.

Retrieves historical Amazon customer queries and their corresponding
agent resolutions to ground LLM-generated draft responses.
"""

import logging

from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from app.config import settings
from app.models import RetrievedExample

log = logging.getLogger(__name__)


class SupportRetriever:
    """Wrapper around LangChain Chroma vector store for Amazon support knowledge."""

    def __init__(
        self, persist_dir: str | None = None, collection_name: str = "amazon_support"
    ) -> None:
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self.collection_name = collection_name
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=settings.embedding_model,
            google_api_key=settings.google_api_key,
        )
        self._vectorstore: Chroma | None = None

    @property
    def vectorstore(self) -> Chroma:
        """Lazy-loaded Chroma vector store."""
        if self._vectorstore is None:
            log.info(
                "Connecting to ChromaDB at %s (collection='%s')",
                self.persist_dir,
                self.collection_name,
            )
            self._vectorstore = Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_dir,
            )
        return self._vectorstore

    def retrieve_similar(
        self, query: str, k: int | None = None
    ) -> list[RetrievedExample]:
        """
        Retrieve top-k past customer conversations similar to the incoming query.
        Returns a list of RetrievedExample objects with customer query and historical agent reply.
        """
        top_k = k or settings.max_retrieved_examples
        try:
            results = self.vectorstore.similarity_search_with_relevance_scores(
                query, k=top_k
            )
            examples: list[RetrievedExample] = []
            for doc, score in results:
                agent_reply = doc.metadata.get("agent_reply", "")
                examples.append(
                    RetrievedExample(
                        customer_message=doc.page_content,
                        agent_reply=agent_reply,
                        score=round(float(score), 4) if score is not None else None,
                    )
                )
            return examples
        except Exception as e:
            log.warning(
                "Vector retrieval failed or collection empty: %s. Returning fallback examples.",
                e,
            )
            return []


# Global singleton instance
retriever = SupportRetriever()
