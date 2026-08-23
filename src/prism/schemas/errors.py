"""OpenAI-shaped error envelope, plus Prism's internal error taxonomy.

The taxonomy is the point. ``429`` (you exceeded your quota — back off, respect
``retry-after``) and ``529`` (the provider is over capacity — your quota is
irrelevant, fail over) demand opposite responses, so they are separate members
here and stay separate all the way into the trace table. Only ``UPSTREAM_OVERLOADED``
and transport failures should ever count toward opening a circuit breaker.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

from pydantic import BaseModel


class ErrorKind(StrEnum):
    INVALID_REQUEST = "invalid_request_error"  # 400 from the client
    TRANSLATION = "translation_error"  # we could not express it upstream
    AUTHENTICATION = "authentication_error"  # 401 at the gateway
    UPSTREAM_INVALID_REQUEST = "upstream_invalid_request"  # 400 from the provider
    UPSTREAM_AUTH = "upstream_authentication_error"  # 401/403 from the provider
    UPSTREAM_NOT_FOUND = "upstream_not_found"  # 404 from the provider
    UPSTREAM_RATE_LIMIT = "upstream_rate_limit"  # 429 — quota exhausted
    UPSTREAM_OVERLOADED = "upstream_overloaded"  # 529 — provider capacity
    UPSTREAM_SERVER_ERROR = "upstream_server_error"  # 5xx other than 529
    UPSTREAM_TIMEOUT = "upstream_timeout"
    UPSTREAM_CONNECTION = "upstream_connection_error"
    INTERNAL = "internal_error"

    @property
    def counts_toward_circuit_breaker(self) -> bool:
        return self in {
            ErrorKind.UPSTREAM_OVERLOADED,
            ErrorKind.UPSTREAM_SERVER_ERROR,
            ErrorKind.UPSTREAM_TIMEOUT,
            ErrorKind.UPSTREAM_CONNECTION,
        }

    @property
    def is_retryable(self) -> bool:
        return self in {
            ErrorKind.UPSTREAM_RATE_LIMIT,
            ErrorKind.UPSTREAM_OVERLOADED,
            ErrorKind.UPSTREAM_SERVER_ERROR,
            ErrorKind.UPSTREAM_TIMEOUT,
            ErrorKind.UPSTREAM_CONNECTION,
        }


class ErrorBody(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody


class PrismError(Exception):
    """Every failure surfaced to a client passes through here."""

    def __init__(
        self,
        kind: ErrorKind,
        message: str,
        *,
        status_code: int = 500,
        param: str | None = None,
        retry_after: float | None = None,
        upstream_status: int | None = None,
        upstream_body: Any = None,
    ) -> None:
        super().__init__(message)
        self.kind = kind
        self.message = message
        self.status_code = status_code
        self.param = param
        self.retry_after = retry_after
        self.upstream_status = upstream_status
        self.upstream_body = upstream_body

    def to_response(self) -> ErrorResponse:
        return ErrorResponse(
            error=ErrorBody(message=self.message, type=str(self.kind), param=self.param)
        )
