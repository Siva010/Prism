"""Data access for auth and traces."""

from __future__ import annotations

import hashlib
import secrets
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from .models import ApiKey, Tenant, Trace

KEY_PREFIX = "prism_sk_"


def hash_key(plaintext: str) -> str:
    return hashlib.sha256(plaintext.encode("utf-8")).hexdigest()


def generate_key() -> tuple[str, str, str]:
    """Return (plaintext, hash, display_prefix). The plaintext is never stored."""
    plaintext = KEY_PREFIX + secrets.token_urlsafe(32)
    return plaintext, hash_key(plaintext), plaintext[: len(KEY_PREFIX) + 6]


@dataclass(frozen=True)
class Principal:
    tenant_id: uuid.UUID
    tenant_slug: str
    api_key_id: uuid.UUID
    scopes: tuple[str, ...]


async def authenticate(session: AsyncSession, presented_key: str) -> Principal | None:
    """Look a key up by hash. Constant-time comparison is the database's index."""
    stmt = (
        select(ApiKey, Tenant)
        .join(Tenant, Tenant.id == ApiKey.tenant_id)
        .where(ApiKey.key_hash == hash_key(presented_key))
        .where(ApiKey.revoked_at.is_(None))
        .where(Tenant.is_active.is_(True))
    )
    row = (await session.execute(stmt)).first()
    if row is None:
        return None
    api_key, tenant = row
    return Principal(
        tenant_id=tenant.id,
        tenant_slug=tenant.slug,
        api_key_id=api_key.id,
        scopes=tuple(api_key.scopes or ()),
    )


async def touch_key(session: AsyncSession, api_key_id: uuid.UUID) -> None:
    await session.execute(
        update(ApiKey)
        .where(ApiKey.id == api_key_id)
        .values(last_used_at=datetime.now(UTC))
    )


async def create_tenant_with_key(
    session: AsyncSession, slug: str, name: str, label: str = "default"
) -> tuple[Tenant, str]:
    tenant = Tenant(id=uuid.uuid4(), slug=slug, name=name, is_active=True)
    session.add(tenant)
    await session.flush()

    plaintext, digest, prefix = generate_key()
    session.add(
        ApiKey(
            id=uuid.uuid4(),
            tenant_id=tenant.id,
            key_hash=digest,
            key_prefix=prefix,
            label=label,
            scopes=["chat", "admin"],
        )
    )
    await session.commit()
    return tenant, plaintext


async def insert_trace(session: AsyncSession, values: dict[str, Any]) -> uuid.UUID:
    trace_id = values.get("id") or uuid.uuid4()
    session.add(Trace(**{**values, "id": trace_id}))
    await session.commit()
    return trace_id


async def list_traces(
    session: AsyncSession,
    *,
    tenant_id: uuid.UUID | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Trace]:
    stmt = select(Trace).order_by(Trace.created_at.desc()).limit(limit).offset(offset)
    if tenant_id is not None:
        stmt = stmt.where(Trace.tenant_id == tenant_id)
    return list((await session.execute(stmt)).scalars())


async def get_trace(session: AsyncSession, trace_id: uuid.UUID) -> Trace | None:
    return (
        await session.execute(select(Trace).where(Trace.id == trace_id))
    ).scalar_one_or_none()
