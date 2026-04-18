from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.core.logging import get_logger

logger = get_logger(__name__)

# --------------------------------------------------------------------------------------------
# Base exception
# --------------------------------------------------------------------------------------------

class AppError(Exception):
    """Every custom exception in this app inherits from this."""
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code: str = "internal_error"
    message: str = "An unexpected error occurred."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.__class__.message
        super().__init__(self.message)

# --------------------------------------------------------------------------------------------
# Auth
# --------------------------------------------------------------------------------------------

class UnauthorizedError(AppError):
    status_code = status.HTTP_401_UNAUTHORIZED
    error_code = "unauthorized"
    message = "Missing or invalid API key."

# --------------------------------------------------------------------------------------------
# File
# --------------------------------------------------------------------------------------------

class UnsupportedMediaTypeError(AppError):
    status_code = status.HTTP_415_UNSUPPORTED_MEDIA_TYPE
    error_code = "unsupported_media_type"
    message = "The uploaded file type is not supported."

class FileTooLargeError(AppError):
    status_code = status.HTTP_413_CONTENT_TOO_LARGE
    error_code = "file_too_large"
    message = "The uploaded file exceeds the maximum allowed size."

class EmptyFileError(AppError):
    status_code = status.HTTP_400_BAD_REQUEST
    error_code = "empty_file"
    message = "The uploaded file is empty."

# --------------------------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------------------------

class DetectionError(AppError):
    status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
    error_code = "detection_error"
    message = "Detection pipeline failed."


class ModelNotReadyError(AppError):
    status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    error_code = "model_not_ready"
    message = "The detection model is not yet loaded. Please try again shortly."


# --------------------------------------------------------------------------------------------
# Jobs
# --------------------------------------------------------------------------------------------

class JobNotFoundError(AppError):
    status_code = status.HTTP_404_NOT_FOUND
    error_code = "job_not_found"
    message = "No job found with the given ID."

# --------------------------------------------------------------------------------------------
# Handlers — what FastAPI actually calls when an exception is raised
# --------------------------------------------------------------------------------------------
async def _app_error_handler(request: Request, exc: AppError) -> JSONResponse:
    logger.warning("app_error code=%s path=%s message=%s",
                   exc.error_code, request.url.path, exc.message)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.error_code, "message": exc.message},
    )

async def _validation_error_handler(request: Request, exc: RequestValidationError) -> JSONResponse:
    logger.warning("validation_error  path=%s  errors=%s", request.url.path, exc.errors())
    return JSONResponse(
        status_code = status.HTTP_422_UNPROCESSABLE_CONTENT,
        content={
            "error": "validation_error",
            "message": "Request validation failed.",
            "detail": exc.errors(),
        },
    )

async def _unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception at %s", request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"error": "internal_error", "message": "An unexpected error occurred."},
    )

# Specific order matters here - more specific exceptions should be registered before more general ones.
# If you put exception first, then it would swallow all exceptions, including the ones you want to handle differently.
def register_exception_handlers(app: FastAPI) -> None:
    """Call this in main.py after creating the FastAPI instance."""
    app.add_exception_handler(RequestValidationError, _validation_error_handler)
    app.add_exception_handler(AppError, _app_error_handler)   
    app.add_exception_handler(Exception, _unhandled_error_handler)

"""
How to use: In any service or route file, you raise the right exception.

Example:

from app.common.exceptions import FileTooLargeError, UnsupportedMediaTypeError

if file.size > settings.max_image_size_bytes:
    raise FileTooLargeError()

if file.content_type not in settings.allowed_image_types:
    raise UnsupportedMediaTypeError()

"""