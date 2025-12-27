"""Query endpoints for RAG Q&A."""

import time

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from app.api.schemas import (
    ErrorResponse,
    EvaluationScores,
    QueryRequest,
    QueryResponse,
    SourceDocument,
)
from app.core.rag_chain import RAGChain
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/query", tags=["Query"])


@router.post(
    "",
    response_model=QueryResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Query processing error"},
    },
    summary="Ask a question",
    description="Submit a question and get an AI-generated answer based on the ingested documents.",
)
async def query(request: QueryRequest) -> QueryResponse:
    """Process a RAG query."""
    logger.info(
        f"Query received: {request.question[:100]}... "
        f"(sources={request.include_sources}, eval={request.enable_evaluation})"
    )
    start_time = time.time()

    try:
        rag_chain = RAGChain()

        # Determine which method to call based on request
        if request.enable_evaluation:
            # Evaluation requires sources, so we always include them
            result = await rag_chain.aquery_with_evaluation(
                question=request.question,
                include_sources=request.include_sources,
            )

            sources = (
                [
                    SourceDocument(
                        content=source["content"],
                        metadata=source["metadata"],
                    )
                    for source in result["sources"]
                ]
                if request.include_sources
                else None
            )

            answer = result["answer"]
            evaluation = EvaluationScores(**result["evaluation"])

        elif request.include_sources:
            result = await rag_chain.aquery_with_sources(request.question)
            sources = [
                SourceDocument(
                    content=source["content"],
                    metadata=source["metadata"],
                )
                for source in result["sources"]
            ]
            answer = result["answer"]
            evaluation = None
        else:
            answer = await rag_chain.aquery(request.question)
            sources = None
            evaluation = None

        processing_time = (time.time() - start_time) * 1000

        logger.info(
            f"Query processed in {processing_time:.2f}ms "
            f"(eval_included={request.enable_evaluation})"
        )

        return QueryResponse(
            question=request.question,
            answer=answer,
            sources=sources,
            processing_time_ms=round(processing_time, 2),
            evaluation=evaluation,
        )

    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@router.post(
    "/stream",
    responses={
        400: {"model": ErrorResponse, "description": "Invalid request"},
        500: {"model": ErrorResponse, "description": "Query processing error"},
    },
    summary="Ask a question (streaming)",
    description="Submit a question and get a streaming AI-generated answer.",
)
async def query_stream(request: QueryRequest) -> StreamingResponse:
    """Process a RAG query with streaming response."""
    logger.info(f"Streaming query received: {request.question[:100]}...")

    try:
        rag_chain = RAGChain()

        async def generate():
            """Generate streaming response."""
            try:
                for chunk in rag_chain.stream(request.question):
                    yield chunk
            except Exception as e:
                logger.error(f"Error in stream: {e}")
                yield f"\n\nError: {str(e)}"

        return StreamingResponse(
            generate(),
            media_type="text/plain",
        )

    except Exception as e:
        logger.error(f"Error setting up stream: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing query: {str(e)}",
        )


@router.post(
    "/search",
    responses={
        500: {"model": ErrorResponse, "description": "Search error"},
    },
    summary="Search documents",
    description="Search for relevant documents without generating an answer.",
)
async def search_documents(
    request: QueryRequest,
) -> dict:
    """Search for relevant documents."""
    logger.info(f"Search received: {request.question[:100]}...")

    try:
        from app.core.vector_store import VectorStoreService

        vector_store = VectorStoreService()
        results = vector_store.search_with_scores(request.question)

        documents = [
            {
                "content": doc.page_content,
                "metadata": doc.metadata,
                "relevance_score": round(score, 4),
            }
            for doc, score in results
        ]

        return {
            "query": request.question,
            "results": documents,
            "count": len(documents),
        }

    except Exception as e:
        logger.error(f"Error searching documents: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error searching documents: {str(e)}",
        )