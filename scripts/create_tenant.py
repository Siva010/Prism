"""Create a tenant and mint its first API key.

    python scripts/create_tenant.py acme "Acme Corp"

The plaintext key is printed once and never stored — only its SHA-256 is.
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from prism.config import get_settings  # noqa: E402
from prism.db.engine import dispose_engine, init_engine, session_scope  # noqa: E402
from prism.db.repo import create_tenant_with_key  # noqa: E402


async def main(slug: str, name: str) -> int:
    init_engine(get_settings().database_url)
    async with session_scope() as session:
        tenant, plaintext = await create_tenant_with_key(session, slug, name)
    await dispose_engine()

    print(f"tenant  {tenant.slug}  ({tenant.id})")
    print(f"api key {plaintext}")
    print("\nStore it now — this is the only time it is shown.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(2)
    slug = sys.argv[1]
    raise SystemExit(asyncio.run(main(slug, sys.argv[2] if len(sys.argv) > 2 else slug)))
