from fastapi import APIRouter, Depends, UploadFile, File
from fastapi import status

from app.core.config import Settings, get_settings
from app.core.logging import get_logger
from app.common.auth import require_api_key
from app.common.responses import SuccessResponse, ok
from app.common.exceptions import UnsupportedMediaTypeError, FileTooLargeError, EmptyFileError
from app.features.image.schemas import ImageDetectionResult
from app.features.image.service import ImageDetectionService
from app.main import image_service as service

logger = get_logger(__name__)
router = APIRouter()

@router.post(
    "/",
    response_model=SuccessResponse[ImageDetectionResult],
    status_code=status.HTTP_200_OK,
    dependencies=[Depends(require_api_key)],
    summary="Detect if an image is AI-generated",
)

async def detect_image(
    file: UploadFile = File(..., description="Image file to analyze"),
    settings: Settings = Depends(get_settings),
):
    # --- Validate MIME type ---
    if file.content_type not in settings.allowed_image_types:
        raise UnsupportedMediaTypeError()
    
    # --- Read file into memory ---
    contents = await file.read()

    # --- Validate size ---
    if len(contents) == 0:
        raise EmptyFileError()
    
    if len(contents) > settings.max_image_size_bytes:
        raise FileTooLargeError()
    
    logger.info("image_detection_request filename=%s size=%d type=%s", file.filename, len(contents), file.content_type)

    # --- Run detection ---
    result = await service.analyze(contents)

    return ok(result)
