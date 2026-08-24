"""FastAPI application factory."""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .api import admin_routes, openai_routes
from .caching.embeddings import build_embedder
from .caching.semantic import SemanticCache, SemanticCacheConfig
from .caching.store import PgVectorStore
from .config import Settings, get_settings
from .db.engine import dispose_engine, get_sessionmaker, init_engine
from .logging_setup import configure_logging, get_logger
from .providers.anthropic_provider import AnthropicProvider
from .routing.economics import EscalationMode
from .routing.model import DifficultyRouter
from .routing.policy import RouterPolicy
from .schemas.errors import ErrorBody, ErrorKind, ErrorResponse, PrismError
from .tracing.recorder import TraceRecorder

log = get_logger(__name__)


def _build_router(settings: Settings) -> RouterPolicy | None:
    """Load the trained router, or None.

    A missing or unloadable model file is logged and tolerated rather than fatal:
    `RouterPolicy` with no router defaults to the top of the ladder, so the
    failure mode is "spends more than it needed to" rather than "silently serves
    worse answers".
    """
    if not settings.router_enabled:
        return None
    router = None
    try:
        router = DifficultyRouter.load(settings.router_model_path)
    except Exception as exc:  # noqa: BLE001
        log.warning(
            "router_model_unavailable",
            path=settings.router_model_path,
            error=str(exc),
            effect="routing every request to the top of the ladder",
        )
    return RouterPolicy(
        router=router,
        mode=EscalationMode(settings.router_mode),
        verifier_fraction=settings.router_verifier_fraction,
        quality_floor=settings.router_quality_floor,
    )


def _build_semantic_cache(settings: Settings) -> SemanticCache | None:
    """Construct layer 2, or None when it is switched off.

    The embedder is built even when the cache is disabled only if it is cheap to
    do so; `LocalEmbedder` loads its weights lazily, so this costs nothing until
    the first lookup.
    """
    if not settings.semantic_cache_enabled:
        return None
    embedder = (
        build_embedder(settings.embedder, model_name=settings.embedder_model)
        if settings.embedder == "local"
        else build_embedder(settings.embedder)
    )
    store = PgVectorStore(get_sessionmaker(), ef_search=settings.hnsw_ef_search)
    return SemanticCache(
        store,
        embedder,
        SemanticCacheConfig(
            enabled=True,
            threshold=settings.semantic_cache_threshold,
            min_query_chars=settings.semantic_cache_min_query_chars,
        ),
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    init_engine(settings.database_url)
    app.state.provider = AnthropicProvider(
        settings.anthropic_api_key,
        base_url=settings.anthropic_base_url,
        timeout_s=settings.request_timeout_s,
        connect_timeout_s=settings.connect_timeout_s,
    )
    app.state.recorder = TraceRecorder(include_bodies=settings.trace_bodies)
    app.state.semantic_cache = _build_semantic_cache(settings)
    app.state.router_policy = _build_router(settings)
    log.info(
        "prism_started",
        default_model=settings.default_model,
        prefix_cache=settings.cache_enabled,
        semantic_cache=settings.semantic_cache_enabled,
        router=settings.router_enabled,
    )
    try:
        yield
    finally:
        await app.state.recorder.drain()
        await app.state.provider.aclose()
        await dispose_engine()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    app = FastAPI(
        title="Prism",
        version="0.1.0",
        description="Claude-native LLM gateway.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    # Bind the settings this app was constructed with, so a test or an
    # embedder can pass a Settings object and have request-scoped
    # dependencies see it rather than the process environment.
    app.dependency_overrides[get_settings] = lambda: settings

    @app.middleware("http")
    async def bind_request_id(request: Request, call_next):  # type: ignore[no-untyped-def]
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex
        structlog.contextvars.bind_contextvars(request_id=request_id)
        try:
            response = await call_next(request)
        finally:
            structlog.contextvars.clear_contextvars()
        response.headers["x-request-id"] = request_id
        return response

    @app.exception_handler(PrismError)
    async def prism_error_handler(_: Request, exc: PrismError) -> JSONResponse:
        headers = {}
        if exc.retry_after is not None:
            # Relayed verbatim so a client backs off for as long as the provider
            # asked, rather than guessing.
            headers["retry-after"] = str(int(exc.retry_after))
        if exc.upstream_status is not None:
            headers["x-prism-upstream-status"] = str(exc.upstream_status)
        return JSONResponse(
            exc.to_response().model_dump(),
            status_code=exc.status_code,
            headers=headers,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
        # FastAPI's default 422 body is not the shape an OpenAI client expects.
        first = exc.errors()[0] if exc.errors() else {}
        param = ".".join(str(p) for p in first.get("loc", ())[1:]) or None
        return JSONResponse(
            ErrorResponse(
                error=ErrorBody(
                    message=first.get("msg", "Invalid request."),
                    type=str(ErrorKind.INVALID_REQUEST),
                    param=param,
                )
            ).model_dump(),
            status_code=400,
        )

    @app.get("/healthz")
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    app.include_router(openai_routes.router, prefix="/v1", tags=["openai"])
    app.include_router(admin_routes.router, prefix="/admin", tags=["admin"])
    return app


app = create_app()
