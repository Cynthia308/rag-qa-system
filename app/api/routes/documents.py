"""Document management endpoints."""

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.api.schemas import (
    DocumentListResponse,
    DocumentUploadResponse,
    ErrorResponse,
)
from app.core.document_processor import DocumentProcessor
from app.core.vector_store import VectorStoreService
from app.utils.logger import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post(
    "/upload",
    response_model=DocumentUploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file type"},
        500: {"model": ErrorResponse, "description": "Processing error"},
    },
    summary="Upload and ingest a document",
    description="Upload a document (PDF, TXT, or CSV) to be processed and added to the vector store.",
)
async def upload_document(
    file: UploadFile = File(..., description="Document file to upload"),
) -> DocumentUploadResponse:
    """Upload and process a document."""
    logger.info(f"Received document upload: {file.filename}")

    # Validate file
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )

    try:
        # Process document
        processor = DocumentProcessor()
        chunks = processor.process_upload(file.file, file.filename)

        if not chunks:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the document",
            )

        # Add to vector store
        vector_store = VectorStoreService()
        document_ids = vector_store.add_documents(chunks)

        logger.info(
            f"Successfully processed {file.filename}: "
            f"{len(chunks)} chunks, {len(document_ids)} documents"
        )

        return DocumentUploadResponse(
            message="Document uploaded and processed successfully",
            filename=file.filename,
            chunks_created=len(chunks),
            document_ids=document_ids,
        )

    except ValueError as e:
        logger.warning(f"Invalid file upload: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing document: {str(e)}",
        )


@router.get(
    "/info",
    response_model=DocumentListResponse,
    summary="Get collection information",
    description="Get information about the document collection.",
)
async def get_collection_info() -> DocumentListResponse:
    """Get information about the document collection."""
    logger.debug("Collection info requested")

    try:
        vector_store = VectorStoreService()
        info = vector_store.get_collection_info()

        return DocumentListResponse(
            collection_name=info["name"],
            total_documents=info["points_count"],
            status=info["status"],
        )
    except Exception as e:
        logger.error(f"Error getting collection info: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error getting collection info: {str(e)}",
        )


@router.delete(
    "/collection",
    responses={
        200: {"description": "Collection deleted successfully"},
        500: {"model": ErrorResponse, "description": "Deletion error"},
    },
    summary="Delete the entire collection",
    description="Delete all documents from the vector store. Use with caution!",
)
async def delete_collection() -> dict:
    """Delete the entire document collection."""
    logger.warning("Collection deletion requested")

    try:
        vector_store = VectorStoreService()
        vector_store.delete_collection()

        return {"message": "Collection deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting collection: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting collection: {str(e)}",
        )