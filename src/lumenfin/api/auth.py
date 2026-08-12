"""API key authentication and tenant authorization.

Maps credentials to an AuthenticatedPrincipal. Callers may not impersonate
another tenant by passing a different tenant_id.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from fastapi import Header, HTTPException, status


@dataclass(frozen=True)
class AuthenticatedPrincipal:
    """Server-side identity derived from the API key (not from request body)."""

    client_id: str
    tenant_id: str


def _parse_principals_json(raw: str | None) -> dict[str, AuthenticatedPrincipal]:
    if not raw or not raw.strip():
        return {}
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("MAS_API_KEY_PRINCIPALS must be valid JSON") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("MAS_API_KEY_PRINCIPALS must be a JSON object")
    principals: dict[str, AuthenticatedPrincipal] = {}
    for key, value in payload.items():
        api_key = str(key).strip()
        if not api_key:
            continue
        if isinstance(value, str):
            tenant_id = value.strip() or "default"
            client_id = f"client-{tenant_id}"
        elif isinstance(value, dict):
            tenant_id = str(value.get("tenant_id") or "").strip() or "default"
            client_id = str(value.get("client_id") or f"client-{tenant_id}").strip() or f"client-{tenant_id}"
        else:
            raise RuntimeError(
                f"MAS_API_KEY_PRINCIPALS entry for key {api_key!r} must be a string tenant_id "
                "or an object with tenant_id/client_id"
            )
        principals[api_key] = AuthenticatedPrincipal(client_id=client_id, tenant_id=tenant_id)
    return principals


def build_principal_directory(
    *,
    legacy_api_key: str | None,
    legacy_client_id: str,
    legacy_tenant_id: str,
    principals_json: str | None,
) -> dict[str, AuthenticatedPrincipal]:
    """Merge multi-key JSON map with optional legacy single MAS_API_KEY binding."""
    principals = _parse_principals_json(principals_json)
    legacy_key = (legacy_api_key or "").strip()
    if legacy_key and legacy_key not in principals:
        principals[legacy_key] = AuthenticatedPrincipal(
            client_id=(legacy_client_id or "default-client").strip() or "default-client",
            tenant_id=(legacy_tenant_id or "default").strip() or "default",
        )
    return principals


def resolve_effective_tenant(
    principal: AuthenticatedPrincipal,
    requested_tenant_id: str | None,
) -> str:
    """Authorize a request-scoped tenant claim against the authenticated principal.

    - Missing / blank request tenant → principal.tenant_id
    - Explicit match → ok
    - Explicit mismatch → HTTP 403 (do not silently rewrite)
    """
    if requested_tenant_id is None:
        return principal.tenant_id
    requested = requested_tenant_id.strip()
    if not requested:
        return principal.tenant_id
    if requested != principal.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="tenant_id does not match authenticated principal",
        )
    return requested


def build_api_key_dependency(
    expected_api_key: Optional[str] = None,
    *,
    require_key: bool = False,
    principals: dict[str, AuthenticatedPrincipal] | None = None,
    anonymous_principal: AuthenticatedPrincipal | None = None,
):
    """FastAPI dependency returning AuthenticatedPrincipal.

    Backward compatible with a single MAS_API_KEY: pass it via ``principals``
    or the legacy ``expected_api_key`` argument (mapped into ``principals``).
    """

    directory = dict(principals or {})
    legacy = (expected_api_key or "").strip()
    if legacy and legacy not in directory:
        anon = anonymous_principal or AuthenticatedPrincipal(
            client_id="default-client",
            tenant_id="default",
        )
        directory[legacy] = AuthenticatedPrincipal(
            client_id=anon.client_id,
            tenant_id=anon.tenant_id,
        )
    default_principal = anonymous_principal or AuthenticatedPrincipal(
        client_id="anonymous",
        tenant_id="default",
    )

    def require_api_key(
        x_api_key: Optional[str] = Header(default=None),
    ) -> AuthenticatedPrincipal:
        if not directory:
            if require_key:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="MAS_API_KEY is required when APP_ENV is not dev/test.",
                )
            return default_principal
        presented = (x_api_key or "").strip()
        principal = directory.get(presented)
        if principal is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or missing X-API-Key.",
            )
        return principal

    return require_api_key
