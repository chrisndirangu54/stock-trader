"""Synchronized clocks, margin models and cross-asset risk budgeting."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize


@dataclass(frozen=True)
class MarginRule:
    initial: float
    maintenance: float
    max_leverage: float


DEFAULT_MARGIN_RULES: Dict[str, MarginRule] = {
    "equity": MarginRule(0.50, 0.25, 2.0),
    "crypto": MarginRule(1.00, 1.00, 1.0),
    "forex": MarginRule(0.0333, 0.025, 30.0),
    "future": MarginRule(0.12, 0.10, 8.0),
    "commodity_future": MarginRule(0.12, 0.10, 8.0),
}

ASSET_CLASS_BUDGETS = {
    "equity": 0.35, "crypto": 0.15, "forex": 0.25,
    "future": 0.25, "commodity_future": 0.25,
}


class MarketClock:
    """Normalize heterogeneous sessions to a UTC decision grid."""
    def __init__(self, decision_hour_utc: int = 21):
        self.decision_hour_utc = decision_hour_utc

    @staticmethod
    def is_open(asset_class: str, ts: pd.Timestamp) -> bool:
        ts = pd.Timestamp(ts, tz="UTC") if pd.Timestamp(ts).tzinfo is None else pd.Timestamp(ts).tz_convert("UTC")
        wd = ts.weekday()
        if asset_class == "crypto":
            return True
        if asset_class == "forex":
            if wd == 5:
                return False
            if wd == 6:
                return ts.hour >= 22
            if wd == 4:
                return ts.hour < 22
            return True
        if asset_class == "equity":
            return wd < 5 and 14 <= ts.hour <= 21
        if asset_class in {"future", "commodity_future"}:
            return wd < 5 or (wd == 6 and ts.hour >= 22)
        return wd < 5

    def decision_index(self, start, end) -> pd.DatetimeIndex:
        days = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end).normalize(), freq="D", tz="UTC")
        return days + pd.Timedelta(hours=self.decision_hour_utc)

    def synchronize(self, series: Dict[str, pd.Series], asset_classes: Dict[str, str],
                    tolerance: str = "36h") -> pd.DataFrame:
        start = max(pd.to_datetime(s.index).min() for s in series.values())
        end = min(pd.to_datetime(s.index).max() for s in series.values())
        grid = self.decision_index(start, end)
        out, tol = {}, pd.Timedelta(tolerance)
        for symbol, s in series.items():
            x = s.copy()
            x.index = pd.to_datetime(x.index, utc=True)
            x = x.sort_index()
            aligned = x.reindex(grid, method="ffill", tolerance=tol)
            mask = [self.is_open(asset_classes[symbol], t) or asset_classes[symbol] == "crypto" for t in grid]
            out[symbol] = aligned.where(mask)
        return pd.DataFrame(out).dropna(how="all")


def portfolio_margin(weights: pd.Series, asset_classes: Dict[str, str],
                     rules: Dict[str, MarginRule] | None = None) -> float:
    rules = rules or DEFAULT_MARGIN_RULES
    return sum(abs(float(w)) * rules[asset_classes[s]].initial for s, w in weights.items())


def enforce_margin(weights: pd.Series, asset_classes: Dict[str, str],
                   max_margin_fraction: float = 0.60) -> pd.Series:
    used = portfolio_margin(weights, asset_classes)
    if used <= max_margin_fraction or used <= 0:
        return weights
    return weights * (max_margin_fraction / used)


def risk_contributions(weights: np.ndarray, cov: np.ndarray) -> np.ndarray:
    port_var = float(weights @ cov @ weights)
    if port_var <= 1e-16:
        return np.zeros_like(weights)
    return weights * (cov @ weights) / math.sqrt(port_var)


def risk_budget_weights(cov: pd.DataFrame, asset_classes: Dict[str, str],
                        class_budgets: Dict[str, float] | None = None,
                        gross: float = 1.0) -> pd.Series:
    """Long-only risk-parity allocator across assets with class-level risk budgets."""
    class_budgets = class_budgets or ASSET_CLASS_BUDGETS
    assets = list(cov.columns)
    sigma = cov.values.astype(float)
    target = np.array([class_budgets.get(asset_classes[a], 0.10) for a in assets], dtype=float)
    for cls in set(asset_classes.values()):
        idx = [i for i, a in enumerate(assets) if asset_classes[a] == cls]
        if idx:
            target[idx] = class_budgets.get(cls, 0.10) / len(idx)
    target = target / target.sum()

    def obj(w):
        rc = np.abs(risk_contributions(w, sigma))
        if rc.sum() <= 1e-12:
            return 1e6
        return float(np.sum((rc / rc.sum() - target) ** 2))

    x0 = np.ones(len(assets)) / len(assets)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0.0, gross)] * len(assets),
                   constraints=[{"type": "eq", "fun": lambda w: np.sum(w) - gross}],
                   options={"maxiter": 500, "ftol": 1e-12})
    return pd.Series(res.x if res.success else x0 * gross, index=assets)


def combine_alpha_with_risk_budget(alpha_weights: pd.Series, cov: pd.DataFrame,
                                   asset_classes: Dict[str, str], gross_cap: float = 1.5) -> pd.Series:
    budget = risk_budget_weights(cov, asset_classes, gross=1.0)
    signed = np.sign(alpha_weights.reindex(budget.index).fillna(0.0)) * budget
    mag = alpha_weights.abs().reindex(budget.index).fillna(0.0)
    if mag.max() > 0:
        mag = 0.25 + 0.75 * mag / mag.max()
    out = signed * mag
    if out.abs().sum() > 0:
        out *= min(gross_cap / out.abs().sum(), 1.0)
    return out


def project_signed_risk_budget(signal_weights: pd.Series, cov: pd.DataFrame, betas: pd.Series,
                               asset_classes: Dict[str, str], gross_cap: float = 1.5,
                               name_cap: float = 0.20) -> pd.Series:
    """Risk-budget magnitudes, then project to dollar/beta neutrality."""
    assets = list(signal_weights.index)
    c = cov.reindex(index=assets, columns=assets).fillna(0.0)
    b = risk_budget_weights(c, asset_classes, gross=min(gross_cap, 1.0))
    target = np.sign(signal_weights.reindex(assets).fillna(0.0).values) * b.values
    beta = betas.reindex(assets).fillna(1.0).values.astype(float)
    sigma = c.values.astype(float)

    def obj(w):
        return float(np.sum((w - target) ** 2) + 0.25 * (w @ sigma @ w))

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w)},
        {"type": "eq", "fun": lambda w: float(beta @ w)},
        {"type": "ineq", "fun": lambda w: gross_cap - np.sum(np.abs(w))},
    ]
    x0 = signal_weights.reindex(assets).fillna(0.0).clip(-name_cap, name_cap).values
    res = minimize(obj, x0, method="SLSQP", bounds=[(-name_cap, name_cap)] * len(assets),
                   constraints=cons, options={"maxiter": 500, "ftol": 1e-12})
    return pd.Series(res.x if res.success else x0, index=assets)
