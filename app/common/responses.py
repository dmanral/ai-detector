from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel

T = TypeVar("T")

# ------------------------------------------------------------------------------------
# Response models - also used as OpenAPI schema in route definitions.
# ------------------------------------------------------------------------------------

class SuccessResponse(BaseModel, Generic[T]):
    success: bool = True
    data: T

class JobAcceptedResponse(BaseModel):
    """Returned immediately when an async detection job is queued."""
    success: bool = True
    job_id: str
    status: str = "queued"
    poll_url: str

# ------------------------------------------------------------------------------------
# Helpers - use these in your route handlers.
# ------------------------------------------------------------------------------------

def ok(data: Any) -> dict[str, Any]:
    """Wrap a payload in the standard success envelope."""
    return {"success": True, "data": data}

def accepted(job_id: str, poll_url: str) -> dict[str, Any]:
    """Return a job accepted envelope for asunc endpoints."""
    return {
        "success": True,
        "job_id": job_id,
        "status": "queued",
        "poll_url": poll_url,
    }