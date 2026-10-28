from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

ASSUMPTIONS = {
    "supplier_lead_time_days": {"FOODS_1": 5, "FOODS_2": 7, "FOODS_3": 10},
    "lead_time_cv": 0.4,
    "dc_to_store_transit_days": 2,
    "store_inventory_cap_days": 21,
    "initial_stock_days": 7,
    "holding_cost_rate_annual": 0.25,
    "lost_sale_cost_rate": 0.35,
    "purchase_order_cost_per_department": 50.0,
    "transfer_cost_per_unit": 0.05,
    "dc_capacity_multiplier": 1.25,
    "store_fill_rate_target": 0.95,
}

DEFAULT_DATA = Path(r"C:\Users\niraj\Downloads\Data")

@dataclass(frozen=True)
class Config:
    sales_path: Path = DEFAULT_DATA / "sales_train_evaluation"
    prices_path: Path = DEFAULT_DATA / "sell_prices"
    calendar_path: Path = DEFAULT_DATA / "calendar"
    output_dir: Path = Path("outputs")
    cutoff: int = 1885
    seed: int = 42
    max_series: int | None = None
    replications: int = 3
    lead_time_rounding: str = "ceil"
    initial_dc_stock_days: int = 14
    policy_service_quantile: float = 0.90
    rolling_folds: tuple[tuple[int, int, int], ...] = ((1801, 1802, 1829), (1829, 1830, 1857), (1857, 1858, 1885))
    stores: tuple[str, ...] = ("CA_1", "CA_2", "CA_3", "CA_4")
    departments: tuple[str, ...] = ("FOODS_1", "FOODS_2", "FOODS_3")
