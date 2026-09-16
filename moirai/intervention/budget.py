"""Hard spend cap. Paid execution fails closed when the cap could be exceeded."""
from __future__ import annotations

from dataclasses import dataclass

from moirai.intervention.schema import CostRecord


class BudgetExceeded(RuntimeError):
    pass


@dataclass(frozen=True)
class PriceTable:
    """USD per million tokens. Defaults: DeepInfra Qwen3-Coder-480B FP4, 2026-09-05 list."""
    input_per_m: float = 0.30
    cached_input_per_m: float = 0.10
    output_per_m: float = 1.00
    worker_per_hour: float = 0.1984
    label: str = "deepinfra-qwen3-coder-480b-fp4-2026-09-05"

    def cost(self, usage: CostRecord) -> float:
        return (
            usage.input_tokens / 1e6 * self.input_per_m
            + usage.cached_input_tokens / 1e6 * self.cached_input_per_m
            + usage.output_tokens / 1e6 * self.output_per_m
            + usage.wall_seconds / 3600.0 * self.worker_per_hour
        )


# High-usage scenario from the infrastructure addendum: 4M cumulative input,
# 30k output, 120 min worker occupancy. Used as the per-trial reservation so
# authorization is conservative before any usage has been measured.
HIGH_SCENARIO = CostRecord(input_tokens=4_000_000, output_tokens=30_000, wall_seconds=120 * 60)


@dataclass
class BudgetGuard:
    cap_usd: float
    spent_usd: float = 0.0
    reserve_per_trial_usd: float | None = None
    in_flight: int = 0
    prices: PriceTable = PriceTable()

    def __post_init__(self) -> None:
        if self.reserve_per_trial_usd is None:
            self.reserve_per_trial_usd = self.prices.cost(HIGH_SCENARIO)

    @property
    def committed_usd(self) -> float:
        return self.spent_usd + self.in_flight * float(self.reserve_per_trial_usd)

    def authorize(self, n: int = 1) -> None:
        """Reserve budget for n trials or raise; never partially authorize."""
        projected = self.committed_usd + n * float(self.reserve_per_trial_usd)
        if projected > self.cap_usd:
            raise BudgetExceeded(
                f"projected spend ${projected:.2f} exceeds cap ${self.cap_usd:.2f} "
                f"(spent ${self.spent_usd:.2f}, in flight {self.in_flight}, reserve ${self.reserve_per_trial_usd:.2f}/trial)"
            )
        self.in_flight += n

    def commit(self, usage: CostRecord) -> float:
        """Record actual cost for one authorized trial; returns cost charged."""
        if self.in_flight <= 0:
            raise RuntimeError("commit without authorization")
        self.in_flight -= 1
        cost = usage.cost_usd if usage.cost_usd > 0 else self.prices.cost(usage)
        self.spent_usd += cost
        return cost

    def release(self) -> None:
        if self.in_flight > 0:
            self.in_flight -= 1

    @property
    def remaining_usd(self) -> float:
        return self.cap_usd - self.committed_usd
