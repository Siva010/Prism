"""Request-scoped dependencies: auth and shared singletons."""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, Header, Request

from ..config import Settings, get_settings
from ..db import repo
from ..db.engine import session_scope
from ..db.repo import Principal
from ..providers.base import Provider
from ..schemas.errors import ErrorKind, PrismError
from ..tracing.recorder import TraceRecorder

# A tenant used only when auth is disabled for local development, so the request
# path is identical either way and traces still have a stable grouping key.
ANONYMOUS = Principal(
    tenant_id=None,  # type: ignore[arg-type]
    tenant_slug="local",
    api_key_id=None,  # type: ignore[arg-type]
    scopes=("chat", "admin"),
)


def get_provider(request: Request) -> Provider:
    return request.app.state.provider  # type: ignore[no-any-return]


def get_recorder(request: Request) -> TraceRecorder:
    return request.app.state.recorder  # type: ignore[no-any-return]


def get_semantic_cache(request: Request) -> Any:
    """Layer 2, or None when disabled. Callers must handle None."""
    return getattr(request.app.state, "semantic_cache", None)


def get_router_policy(request: Request) -> Any:
    """The two-axis router, or None when disabled."""
    return getattr(request.app.state, "router_policy", None)


async def get_principal(
    authorization: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,  # type: ignore[assignment]
) -> Principal:
    if settings.auth_disabled:
        return ANONYMOUS

    if not authorization or not authorization.lower().startswith("bearer "):
        raise PrismError(
            ErrorKind.AUTHENTICATION,
            "Missing bearer token. Send your Prism API key as "
            "`Authorization: Bearer prism_sk_...`.",
            status_code=401,
        )

    presented = authorization.split(" ", 1)[1].strip()
    async with session_scope() as session:
        principal = await repo.authenticate(session, presented)
        if principal is None:
            raise PrismError(
                ErrorKind.AUTHENTICATION, "Invalid or revoked API key.", status_code=401
            )
        await repo.touch_key(session, principal.api_key_id)
        await session.commit()
    return principal


PrincipalDep = Annotated[Principal, Depends(get_principal)]
SettingsDep = Annotated[Settings, Depends(get_settings)]
ProviderDep = Annotated[Provider, Depends(get_provider)]
RecorderDep = Annotated[TraceRecorder, Depends(get_recorder)]
SemanticCacheDep = Annotated[Any, Depends(get_semantic_cache)]
RouterDep = Annotated[Any, Depends(get_router_policy)]
