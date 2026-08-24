"""Per-tenant spend caps.

**Two caps, because they answer different questions.** A soft cap warns and keeps
serving; a hard cap refuses. With only one, you must choose between surprising
people with a bill and cutting them off without warning, and both are bad.

**Degrade before you reject.** At the hard cap, dropping to the cheapest tier
usually serves a tenant better than a 402 — they keep working at a fraction of
the cost, and the operator stops absorbing spend. Rejecting is still available
because some deployments genuinely need a hard stop, but it is a choice rather
than the only option.

**The running total is a cache, not the truth.** `traces` holds every priced
token, so the budget row is a denormalization to keep the hot path from
aggregating on every request. `reconcile` rebuilds it from traces, which means a
crashed process or a lost update costs accuracy until the next reconcile rather
than corrupting the ledger permanently.

Spend is checked *before* dispatch and recorded *after*, so a tenant can overrun
by at most the cost of the requests already in flight. Making that window zero
would mean holding a lock across an upstream call, which trades a bounded
overrun for an unbounded latency problem.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from enum import StrEnum
from typing import Any
from uuid import UUID

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


class BudgetStatus(StrEnum):
    OK = "ok"
    SOFT_CAP = "soft_cap"  # over the warning line, still serving
    HARD_CAP_DEGRADE = "degrade"  # over the hard cap, drop to the cheapest tier
    HARD_CAP_REJECT = "reject"  # over the hard cap, refuse


@dataclass(frozen=True)
class BudgetDecision:
    status: BudgetStatus
    spent_usd: Decimal
    soft_cap_usd: Decimal | None
    hard_cap_usd: Decimal | None
    period_start: date | None = None

    @property
    def allowed(self) -> bool:
        return self.status is not BudgetStatus.HARD_CAP_REJECT

    @property
    def must_degrade(self) -> bool:
        return self.status is BudgetStatus.HARD_CAP_DEGRADE

    @property
    def utilisation(self) -> float | None:
        if not self.hard_cap_usd:
            return None
        return float(self.spent_usd / self.hard_cap_usd)

    def as_json(self) -> dict[str, Any]:
        return {
            "status": str(self.status),
            "spent_usd": str(self.spent_usd),
            "soft_cap_usd": str(self.soft_cap_usd) if self.soft_cap_usd else None,
            "hard_cap_usd": str(self.hard_cap_usd) if self.hard_cap_usd else None,
            "utilisation": self.utilisation,
        }


UNLIMITED = BudgetDecision(BudgetStatus.OK, Decimal(0), None, None)


def _period_start(period: str, today: date | None = None) -> date:
    today = today or datetime.now(UTC).date()
    return today.replace(day=1) if period == "month" else today


async def check(
    session: AsyncSession, tenant_id: UUID | None, *, today: date | None = None
) -> BudgetDecision:
    """Where this tenant stands, before dispatch."""
    if tenant_id is None:
        return UNLIMITED

    row = (
        (
            await session.execute(
                text("""
                SELECT soft_cap_usd, hard_cap_usd, period, period_start, spent_usd,
                       hard_cap_action
                FROM tenant_budgets WHERE tenant_id = :t
            """),
                {"t": str(tenant_id)},
            )
        )
        .mappings()
        .first()
    )

    if row is None:
        return UNLIMITED

    current = _period_start(row["period"], today)
    spent = Decimal(row["spent_usd"] or 0)
    if row["period_start"] != current:
        # The window rolled over. Treating the stale total as current would
        # keep a tenant capped into a period they have not spent anything in.
        spent = Decimal(0)

    hard = Decimal(row["hard_cap_usd"]) if row["hard_cap_usd"] is not None else None
    soft = Decimal(row["soft_cap_usd"]) if row["soft_cap_usd"] is not None else None

    if hard is not None and spent >= hard:
        status = (
            BudgetStatus.HARD_CAP_DEGRADE
            if row["hard_cap_action"] == "degrade"
            else BudgetStatus.HARD_CAP_REJECT
        )
    elif soft is not None and spent >= soft:
        status = BudgetStatus.SOFT_CAP
    else:
        status = BudgetStatus.OK

    return BudgetDecision(status, spent, soft, hard, current)


async def record_spend(
    session: AsyncSession,
    tenant_id: UUID | None,
    cost_usd: Decimal,
    *,
    today: date | None = None,
) -> None:
    """Add to the running total after a request completes."""
    if tenant_id is None or cost_usd <= 0:
        return

    row = (
        (
            await session.execute(
                text("SELECT period FROM tenant_budgets WHERE tenant_id = :t"),
                {"t": str(tenant_id)},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return

    current = _period_start(row["period"], today)
    await session.execute(
        text("""
            UPDATE tenant_budgets
            SET spent_usd = CASE
                    WHEN period_start = :p THEN spent_usd + :c
                    ELSE :c
                END,
                period_start = :p,
                notified_at = CASE WHEN period_start = :p THEN notified_at ELSE NULL END,
                updated_at = now()
            WHERE tenant_id = :t
        """),
        {"t": str(tenant_id), "c": cost_usd, "p": current},
    )
    await session.commit()


async def record_event(
    session: AsyncSession,
    tenant_id: UUID,
    event: str,
    decision: BudgetDecision,
    detail: dict[str, Any] | None = None,
) -> None:
    """Append to the ledger, so 'why was I cut off?' has an answer."""
    import json

    await session.execute(
        text("""
            INSERT INTO budget_events (tenant_id, event, spent_usd, cap_usd, detail)
            VALUES (:t, :e, :s, :c, CAST(:d AS JSONB))
        """),
        {
            "t": str(tenant_id),
            "e": event,
            "s": decision.spent_usd,
            "c": decision.hard_cap_usd or decision.soft_cap_usd,
            "d": json.dumps(detail or {}),
        },
    )
    await session.commit()


async def reconcile(
    session: AsyncSession, tenant_id: UUID, *, today: date | None = None
) -> Decimal:
    """Rebuild the running total from traces.

    The denormalized figure can drift — a process dies between the upstream call
    and the update, and that spend is recorded on the trace but not the budget.
    Traces are the source of truth precisely so this is recoverable.
    """
    row = (
        (
            await session.execute(
                text("SELECT period FROM tenant_budgets WHERE tenant_id = :t"),
                {"t": str(tenant_id)},
            )
        )
        .mappings()
        .first()
    )
    if row is None:
        return Decimal(0)

    current = _period_start(row["period"], today)
    actual = (
        await session.execute(
            text("""
                SELECT COALESCE(sum(cost_usd), 0) FROM traces
                WHERE tenant_id = :t AND created_at >= :p
            """),
            {"t": str(tenant_id), "p": current},
        )
    ).scalar_one()

    await session.execute(
        text("""
            UPDATE tenant_budgets
            SET spent_usd = :s, period_start = :p, updated_at = now()
            WHERE tenant_id = :t
        """),
        {"t": str(tenant_id), "s": actual, "p": current},
    )
    await session.commit()
    return Decimal(actual)


async def set_budget(
    session: AsyncSession,
    tenant_id: UUID,
    *,
    soft_cap_usd: Decimal | None = None,
    hard_cap_usd: Decimal | None = None,
    period: str = "month",
    hard_cap_action: str = "reject",
) -> None:
    if soft_cap_usd is not None and hard_cap_usd is not None and soft_cap_usd > hard_cap_usd:
        # A soft cap above the hard cap can never fire, so the tenant would be
        # cut off having never been warned.
        raise ValueError(
            f"soft cap {soft_cap_usd} exceeds hard cap {hard_cap_usd}; the warning "
            "would never fire before the cut-off"
        )

    await session.execute(
        text("""
            INSERT INTO tenant_budgets
                (tenant_id, soft_cap_usd, hard_cap_usd, period, period_start,
                 hard_cap_action)
            VALUES (:t, :s, :h, :period, :p, :action)
            ON CONFLICT (tenant_id) DO UPDATE SET
                soft_cap_usd = EXCLUDED.soft_cap_usd,
                hard_cap_usd = EXCLUDED.hard_cap_usd,
                period = EXCLUDED.period,
                hard_cap_action = EXCLUDED.hard_cap_action,
                updated_at = now()
        """),
        {
            "t": str(tenant_id),
            "s": soft_cap_usd,
            "h": hard_cap_usd,
            "period": period,
            "p": _period_start(period),
            "action": hard_cap_action,
        },
    )
    await session.commit()
