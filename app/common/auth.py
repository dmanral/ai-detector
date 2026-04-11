from fastapi import Depends, Security
from fastapi.security import APIKeyHeader

from app.core.config import get_settings, Settings
from app.common.exceptions import UnauthorizedError

# Tells FastAPI to look for this header and show it in the Swagger UI.
_api_keY_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

"""
The auth flow:
Request comes in
→ FastAPI sees the route requires auth
→ Runs require_api_key()
→ Checks X-API-Key header against your valid keys in settings
→ Invalid or missing? Raises UnauthorizedError → request stops, 401 returned
→ Valid? Route handler runs normally
"""

async def require_api_key(
    api_key: str | None = Security(_api_keY_scheme),
    settings: Settings = Depends(get_settings),
) -> str:
    """
    FastAPI dependency that validates the X-API-Key header.

    auto_error=False means FastAPI won't throw its own error if the
    header is missing - we handle that ourselves so the error reasponse stays
    in our standard format.
    """
    if not api_key or api_key not in settings.api_keys_set:
        raise UnauthorizedError()
    return api_key