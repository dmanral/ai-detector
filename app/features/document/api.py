from fastapi import APIRouter, Depends, UploadFile, File, status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.common.auth import require_api_key
from app.common.responses import SuccessResponse, ok
from app.common.exceptions import UnsupportedMediaTypeError, FileTooLargeError, EmptyFileError
from app.features.document.schemas import DocumentDetectionResult
from app.main import document_service as service

logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/",
    response_model=SuccessResponse[DocumentDetectionResult],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    summary="Detect if a document is AI-generated",
)

async def detect_document(
    file: UploadFile = File(..., description="Documentfile to analyse (.txt, .pdf, .docx)"),
    settings: Settings = Depends(get_settings),
):
    # --- Validate MIME type ---
    if file.content_type not in settings.allowed_document_types:
        raise UnsupportedMediaTypeError()
    
    # --- Read file into memory ---
    contents = await file.read()

    # --- Validate size ---
    if len(contents) ==0:
        raise EmptyFileError()
    
    if len(contents) > settings.max_document_size_bytes:
        raise FileTooLargeError()
    
    logger.info(
        "document_detection_request filename=%s size=%d type=%s",
        file.filename, len(contents), file.content_type,
    )

    # --- Run detection ---
    result = await service.analyze(contents, file.content_type)

    return ok(result)