"""RAG chain module using LangChain LCEL."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
# from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from app.config import get_settings
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
settings = get_settings()


# RAG Prompt Template
RAG_PROMPT_TEMPLATE = """You are a helpful assistant. Answer the question based on the provided context.

If you cannot answer the question based on the context, say "I don't have enough information to answer that question."

Do not make up information. Only use the context provided.

Context:
{context}

Question: {question}

Answer:"""


def format_docs(docs: list[Document]) -> str:
    """Format documents into a single context string.

    Args:
        docs: List of Document objects

    Returns:
        Formatted context string
    """
    return "\n\n---\n\n".join(doc.page_content for doc in docs)


class RAGChain:
    """RAG chain for question answering."""

    def __init__(self, vector_store_service: VectorStoreService | None = None):
        """Initialize RAG chain.

        Args:
            vector_store_service: Optional VectorStoreService instance
        """
        self.vector_store = vector_store_service or VectorStoreService()
        self.retriever = self.vector_store.get_retriever()

        # Initialize evaluator (lazy load)
        self._evaluator = None

        # Initialize LLM
        self.llm = ChatGoogleGenerativeAI(
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            google_api_key=settings.google_api_key,
        )

        # Create prompt template
        self.prompt = ChatPromptTemplate.from_template(RAG_PROMPT_TEMPLATE)

        # Build LCEL chain
        self.chain = (
            {
                "context": self.retriever | format_docs,
                "question": RunnablePassthrough(),
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        logger.info(
            f"RAGChain initialized with model={settings.llm_model}, "
            f"retrieval_k={settings.retrieval_k}"
        )

    @property
    def evaluator(self):
        """Get or create RAGAS evaluator instance."""
        if self._evaluator is None:
            from app.core.ragas_evaluator import RAGASEvaluator

            self._evaluator = RAGASEvaluator()
        return self._evaluator

    def query(self, question: str) -> str:
        """Execute a RAG query.

        Args:
            question: User question

        Returns:
            Generated answer
        """
        logger.info(f"Processing query: {question[:100]}...")

        try:
            answer = self.chain.invoke(question)
            logger.info("Query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            raise

    def query_with_sources(self, question: str) -> dict:
        """Execute a RAG query and return sources.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing query with sources: {question[:100]}...")

        try:
            # Get answer
            answer = self.chain.invoke(question)

            # Get source documents
            source_docs = self.retriever.invoke(question)

            # Format sources
            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

            logger.info(f"Query processed with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing query with sources: {e}")
            raise

    async def aquery(self, question: str) -> str:
        """Execute an async RAG query.

        Args:
            question: User question

        Returns:
            Generated answer
        """
        logger.info(f"Processing async query: {question[:100]}...")

        try:
            answer = await self.chain.ainvoke(question)
            logger.info("Async query processed successfully")
            return answer
        except Exception as e:
            logger.error(f"Error processing async query: {e}")
            raise

    async def aquery_with_sources(self, question: str) -> dict:
        """Execute an async RAG query and return sources.

        Args:
            question: User question

        Returns:
            Dictionary with answer and source documents
        """
        logger.info(f"Processing async query with sources: {question[:100]}...")

        try:
            # Get answer
            answer = await self.chain.ainvoke(question)

            # Get source documents (sync operation)
            source_docs = self.retriever.invoke(question)

            # Format sources
            sources = [
                {
                    "content": (
                        doc.page_content[:500] + "..."
                        if len(doc.page_content) > 500
                        else doc.page_content
                    ),
                    "metadata": doc.metadata,
                }
                for doc in source_docs
            ]

            logger.info(f"Async query processed with {len(sources)} sources")

            return {
                "answer": answer,
                "sources": sources,
            }
        except Exception as e:
            logger.error(f"Error processing async query with sources: {e}")
            raise

    async def aquery_with_evaluation(self, question: str, include_sources: bool = True) -> dict:
        """Execute async RAG query with RAGAS evaluation.

        Args:
            question: User question
            include_sources: Whether to include sources in response

        Returns:
            Dictionary with answer, sources, and evaluation scores
        """
        logger.info(f"Processing query with evaluation: {question[:100]}...")

        try:
            # Get answer and sources
            result = await self.aquery_with_sources(question)
            answer = result["answer"]
            sources = result["sources"]

            # Prepare contexts for evaluation
            contexts = [source["content"] for source in sources]

            # Run evaluation with error handling
            try:
                evaluation = await self.evaluator.aevaluate(
                    question=question, answer=answer, contexts=contexts
                )
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={evaluation.get('faithfulness', 'N/A')}, "
                    f"answer_relevancy={evaluation.get('answer_relevancy', 'N/A')}"
                )
            except Exception as e:
                logger.warning(f"Evaluation failed: {e}", exc_info=True)
                evaluation = {
                    "faithfulness": None,
                    "answer_relevancy": None,
                    "evaluation_time_ms": None,
                    "error": str(e),
                }

            return {"answer": answer, "sources": sources, "evaluation": evaluation}

        except Exception as e:
            logger.error(f"Error in query with evaluation: {e}")
            raise

    def stream(self, question: str):
        """Stream RAG response.

        Args:
            question: User question

        Yields:
            Response chunks
        """
        logger.info(f"Streaming query: {question[:100]}...")

        try:
            for chunk in self.chain.stream(question):
                yield chunk
        except Exception as e:
            logger.error(f"Error streaming query: {e}")
            raise