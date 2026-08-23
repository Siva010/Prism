"""SQLAlchemy models mirroring migrations/0001_init.sql.

The SQL file is the source of truth (it is what the container applies on first
boot); these classes exist for typed reads and writes from the application.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    ARRAY,
    Boolean,
    DateTime,
    ForeignKey,
    Integer,
    Numeric,
    String,
    Text,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    name: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)


class ApiKey(Base):
    __tablename__ = "api_keys"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("tenants.id", ondelete="CASCADE")
    )
    key_hash: Mapped[str] = mapped_column(Text, unique=True)
    key_prefix: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text, default="")
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    api_key_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))

    endpoint: Mapped[str] = mapped_column(Text)
    ingress_format: Mapped[str] = mapped_column(Text, default="openai")
    provider: Mapped[str] = mapped_column(Text, default="anthropic")
    model_requested: Mapped[str] = mapped_column(Text)
    model_resolved: Mapped[str] = mapped_column(Text)
    stream: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[str] = mapped_column(Text)
    http_status: Mapped[int | None] = mapped_column(Integer)
    error_type: Mapped[str | None] = mapped_column(Text)
    error_message: Mapped[str | None] = mapped_column(Text)

    stop_reason: Mapped[str | None] = mapped_column(Text)
    finish_reason: Mapped[str | None] = mapped_column(Text)

    latency_ms: Mapped[int | None] = mapped_column(Integer)
    ttft_ms: Mapped[int | None] = mapped_column(Integer)
    upstream_latency_ms: Mapped[int | None] = mapped_column(Integer)

    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_read_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    cache_creation_input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    thinking_tokens: Mapped[int] = mapped_column(Integer, default=0)

    cost_usd: Mapped[Decimal] = mapped_column(Numeric(14, 8), default=Decimal(0))
    cost_breakdown: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)

    prompt_version: Mapped[str | None] = mapped_column(Text)
    provider_request_id: Mapped[str | None] = mapped_column(String)

    request_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    upstream_request: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    upstream_response: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    response_body: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    extra: Mapped[dict[str, Any]] = mapped_column(JSONB, default=dict)
