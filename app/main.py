from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.common.exceptions import register_exception_handlers
from app.common.auth import require_api_key
from app.features.image.service import ImageDetectionService
from app.features.document.service import DocumentDetectionService

# Boot logging first - before FastAPI initialises
setup_logging()
logger = get_logger(__name__)
settings = get_settings()
image_service = ImageDetectionService()
document_service = DocumentDetectionService()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info(
        "Starting %s v%s env=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    image_service.load_models()
    document_service.load_models()

    # Later: load ML models, connect to Redis, warm up caches etc.
    yield
    # --- Shutdown ---
    logger.info("Shutting down %s", settings.app_name)
    # Later: release model memory, close connections etc.

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ----------------------------------------------------------------------------------
# Exception handlers
# ----------------------------------------------------------------------------------
register_exception_handlers(app)

# ----------------------------------------------------------------------------------
# Core routes
# ----------------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def read_root():
    return {"message": f"Welcome to {settings.app_name}"}

@app.get("/health", tags=["meta"])
async def health_check():
    return {"status": "ok", "version": settings.app_version}

# ---------------------------------------------------------------------------
# Feature routers — uncomment as each feature is built
# ---------------------------------------------------------------------------

from app.features.image.api import router as image_router
from app.features.document.api import router as document_router
# from app.features.video.api import router as video_router


app.include_router(image_router, prefix="/detect/image", tags=["image"])
app.include_router(document_router, prefix="/detect/document", tags=["document"])
# app.include_router(video_router, prefix="/detect/video", tags=["video"])
