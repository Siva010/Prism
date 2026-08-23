"""Apply pending SQL migrations.

`docker compose up` applies migrations/ on first boot via the Postgres entrypoint,
which only fires for an empty data directory. This script is what applies
everything after that. Migrations are written idempotently and recorded in
schema_migrations, so running it twice is a no-op.

    python scripts/migrate.py
"""

from __future__ import annotations

import asyncio
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from sqlalchemy import text  # noqa: E402
from sqlalchemy.ext.asyncio import create_async_engine  # noqa: E402

from prism.config import get_settings  # noqa: E402

MIGRATIONS_DIR = pathlib.Path(__file__).resolve().parents[1] / "migrations"


async def main() -> int:
    settings = get_settings()
    engine = create_async_engine(settings.database_url)

    async with engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TABLE IF NOT EXISTS schema_migrations ("
                "version TEXT PRIMARY KEY, "
                "applied_at TIMESTAMPTZ NOT NULL DEFAULT now())"
            )
        )
        applied = {
            row[0]
            for row in (await conn.execute(text("SELECT version FROM schema_migrations")))
        }

    for path in sorted(MIGRATIONS_DIR.glob("*.sql")):
        version = path.stem
        if version in applied:
            print(f"  skip  {version}")
            continue
        print(f"  apply {version}")
        async with engine.begin() as conn:
            # exec_driver_sql: the migration files contain multiple statements and
            # dollar-quoted bodies that SQLAlchemy's text() would try to bind.
            await conn.exec_driver_sql(path.read_text(encoding="utf-8"))
            await conn.execute(
                text(
                    "INSERT INTO schema_migrations (version) VALUES (:v) "
                    "ON CONFLICT (version) DO NOTHING"
                ),
                {"v": version},
            )

    await engine.dispose()
    print("migrations up to date")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
