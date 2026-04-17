from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, UploadFile, File

from app.core.config import get_settings
from app.core.logging import setup_logging, get_logger
from app.common.exceptions import register_exception_handlers, UnsupportedMediaTypeError
from app.common.auth import require_api_key
from app.common.responses import ok
from app.features.image.service import ImageDetectionService
from app.features.document.service import DocumentDetectionService
from app.agents.orchestrator import Orchestrator

# Boot logging first - before FastAPI initialises
setup_logging()
logger = get_logger(__name__)
settings = get_settings()

image_service    = ImageDetectionService()
document_service = DocumentDetectionService()
orchestrator     = Orchestrator(image_service, document_service)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup ---
    logger.info(
        "Starting %s v%s  env=%s",
        settings.app_name,
        settings.app_version,
        settings.environment,
    )
    image_service.load_models()
    document_service.load_models()
    logger.info("Orchestrator ready")
    yield
    # --- Shutdown ---
    logger.info("Shutting down %s", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ---------------------------------------------------------------------------
# Exception handlers
# ---------------------------------------------------------------------------
register_exception_handlers(app)

# ---------------------------------------------------------------------------
# Core routes
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def read_root():
    return {"message": f"Welcome to {settings.app_name}"}


@app.get("/health", tags=["meta"])
async def health_check():
    return {"status": "ok", "version": settings.app_version}


# ---------------------------------------------------------------------------
# Unified agent endpoint
# ---------------------------------------------------------------------------

@app.post("/detect", tags=["detect"], dependencies=[Depends(require_api_key)])
async def detect(
    file: UploadFile = File(..., description="Image or document to analyse"),
):
    """
    Unified detection endpoint. Automatically routes to the right subagent
    based on file type. Returns signal scores plus LLM reasoning.
    """
    all_types = settings.allowed_image_types + settings.allowed_document_types
    if file.content_type not in all_types:
        raise UnsupportedMediaTypeError()

    contents = await file.read()
    result = await orchestrator.run(contents, file.content_type)
    return ok(result)


# ---------------------------------------------------------------------------
# Feature routers (individual signal endpoints — still available)
# ---------------------------------------------------------------------------

from app.features.image.api import router as image_router
from app.features.document.api import router as document_router

app.include_router(image_router, prefix="/detect/image", tags=["image"])
app.include_router(document_router, prefix="/detect/document", tags=["document"])