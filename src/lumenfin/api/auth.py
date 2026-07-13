from typing import Optional

from fastapi import Header, HTTPException, status


def build_api_key_dependency(expected_api_key: Optional[str], *, require_key: bool = False):
    def require_api_key(x_api_key: Optional[str] = Header(default=None)) -> None:
        if not expected_api_key:
            if require_key:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MAS_API_KEY is required when APP_ENV is not dev/test.",
                )
            return
        if x_api_key != expected_api_key:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key.",
            )

    return require_api_key
