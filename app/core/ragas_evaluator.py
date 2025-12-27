"""RAGAS evaluation module for RAG quality assessment."""

import asyncio
import time
from typing import Any

from datasets import Dataset
# from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from ragas import evaluate
from ragas.metrics import answer_relevancy, faithfulness

from app.config import get_settings
from app.utils.logger import get_logger

logger = get_logger(__name__)


class RAGASEvaluator:
    """Evaluator for RAG responses using RAGAS metrics."""

    def __init__(self):
        """Initialize RAGAS evaluator with metrics and models."""
        logger.info("Initializing RAGAS evaluator")

        # Get settings inside __init__ to allow mocking in tests
        self.settings = get_settings()

        # Use RAGAS-specific LLM settings if provided, otherwise fall back to default
        eval_llm_model = self.settings.ragas_llm_model or self.settings.llm_model
        eval_llm_temperature = (
            self.settings.ragas_llm_temperature
            if self.settings.ragas_llm_temperature is not None
            else self.settings.llm_temperature
        )
        eval_embedding_model = self.settings.ragas_embedding_model or self.settings.embedding_model

        # Initialize LLM for evaluation
        self.llm = ChatGoogleGenerativeAI(
            model=eval_llm_model,
            temperature=eval_llm_temperature,
            google_api_key=self.settings.google_api_key,
        )

        # Initialize embeddings for evaluation
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=eval_embedding_model,
            openai_api_key=self.settings.openai_api_key,
        )

        # Initialize metrics (reference-free only)
        self.metrics = [
            faithfulness,
            answer_relevancy,
        ]

        logger.info(
            f"RAGAS evaluator initialized - "
            f"LLM: {eval_llm_model} (temp={eval_llm_temperature}), "
            f"Embeddings: {eval_embedding_model}, "
            f"Metrics: {[metric.name for metric in self.metrics]}"
        )

    async def aevaluate(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> dict[str, Any]:
        """Execute async RAGAS evaluation.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dictionary with evaluation scores and metadata
        """
        logger.debug(f"Starting evaluation for question: {question[:100]}...")
        start_time = time.time()

        try:
            # Prepare dataset for RAGAS
            dataset = self._prepare_dataset(question, answer, contexts)

            # Run evaluation in thread pool to avoid blocking event loop
            result = await asyncio.to_thread(
                self._evaluate_with_timeout,
                dataset,
            )

            evaluation_time_ms = (time.time() - start_time) * 1000

            # Extract scores
            scores = {
                "faithfulness": float(result["faithfulness"]) if "faithfulness" in result else None,
                "answer_relevancy": (
                    float(result["answer_relevancy"]) if "answer_relevancy" in result else None
                ),
                "evaluation_time_ms": round(evaluation_time_ms, 2),
                "error": None,
            }

            if self.settings.ragas_log_results:
                logger.info(
                    f"Evaluation completed - "
                    f"faithfulness={scores['faithfulness']}, "
                    f"answer_relevancy={scores['answer_relevancy']}, "
                    f"time={scores['evaluation_time_ms']}ms"
                )

            return scores

        except Exception as e:
            logger.warning(f"Evaluation failed: {e}", exc_info=True)
            return self._handle_evaluation_error(e)

    def _prepare_dataset(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> Dataset:
        """Convert RAG output to RAGAS Dataset format.

        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context documents

        Returns:
            Dataset object for RAGAS evaluation
        """
        # RAGAS expects data in specific format
        data = {
            "question": [question],
            "answer": [answer],
            "contexts": [contexts],  # List of lists
        }

        logger.debug(
            f"Prepared dataset with {len(contexts)} contexts " f"for question: {question[:50]}..."
        )

        return Dataset.from_dict(data)

    def _evaluate_with_timeout(self, dataset: Dataset) -> dict[str, Any]:
        """Execute RAGAS evaluation with timeout.

        Args:
            dataset: Prepared RAGAS dataset

        Returns:
            Evaluation results dictionary

        Raises:
            TimeoutError: If evaluation exceeds timeout
        """
        # Note: asyncio.timeout would be ideal, but RAGAS evaluate() is sync
        # For now, we rely on the async wrapper and trust RAGAS to complete
        # In production, consider using signal.alarm or threading.Timer
        result = evaluate(
            dataset,
            metrics=self.metrics,
            llm=self.llm,
            embeddings=self.embeddings,
        )

        # Convert to dictionary and extract scores
        return result.to_pandas().to_dict("records")[0]

    def _handle_evaluation_error(self, error: Exception) -> dict[str, Any]:
        """Return safe fallback scores on error.

        Args:
            error: The exception that occurred

        Returns:
            Dictionary with null scores and error message
        """
        logger.error(f"Returning fallback scores due to error: {error}")

        return {
            "faithfulness": None,
            "answer_relevancy": None,
            "evaluation_time_ms": None,
            "error": str(error),
        }