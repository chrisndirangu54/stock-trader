"""Orchestration layer that applies cross-asset risk budgets and margin to the core stat-arb engine."""
from __future__ import annotations

import math
import pandas as pd

from trading_system_sota import (Config, build_feature_cube, estimate_beta_vector, infer_asset_class,
                                 latest_weights, realized_cost, run_backtest, shrink_cov)
from cross_asset_risk import enforce_margin, project_signed_risk_budget


def _classes(assets):
    out = {}
    for a in assets:
        c = infer_asset_class(a)
        out[a] = "future" if c == "commodity_future" else c
    return out


def latest_institutional_weights(panel, cfg: Config) -> pd.Series:
    raw = latest_weights(panel, cfg)
    _, close, _ = build_feature_cube(panel, cfg)
    cov = shrink_cov(close, cfg.risk_window)
    beta = estimate_beta_vector(close, cfg.beta_window)
    cov_df = pd.DataFrame(cov, index=close.columns, columns=close.columns)
    beta_s = pd.Series(beta, index=close.columns)
    classes = _classes(close.columns)
    if len(set(classes.values())) > 1:
        raw = project_signed_risk_budget(raw, cov_df, beta_s, classes, cfg.gross_cap, cfg.name_cap)
    return enforce_margin(raw, classes)


def run_institutional_backtest(panel, cfg: Config) -> dict:
    base = run_backtest(panel, cfg)
    _, close, dollar_vol = build_feature_cube(panel, cfg)
    ret = close.pct_change()
    classes = _classes(close.columns)
    prev = pd.Series(0.0, index=close.columns)
    weights = pd.DataFrame(0.0, index=base["weights"].index, columns=close.columns)
    pnl = pd.Series(index=weights.index, dtype=float)
    costs = pd.Series(index=weights.index, dtype=float)

    for date in weights.index:
        if date not in close.index or close.index.get_loc(date) >= len(close.index) - 1:
            continue
        raw = base["weights"].loc[date].reindex(close.columns).fillna(0.0)
        hist = close.loc[:date]
        cov = shrink_cov(hist, cfg.risk_window)
        beta = estimate_beta_vector(hist, cfg.beta_window)
        if len(set(classes.values())) > 1:
            raw = project_signed_risk_budget(
                raw,
                pd.DataFrame(cov, index=close.columns, columns=close.columns),
                pd.Series(beta, index=close.columns), classes, cfg.gross_cap, cfg.name_cap,
            )
        w = enforce_margin(raw, classes)
        nxt = close.index[close.index.get_loc(date) + 1]
        next_ret = ret.loc[nxt].reindex(close.columns).fillna(0.0)
        adv = dollar_vol.loc[date].reindex(close.columns).fillna(cfg.min_adv_usd)
        vol20 = ret.loc[:date].tail(20).std() * math.sqrt(252)
        c = realized_cost(w - prev, adv, vol20, cfg)
        weights.loc[date] = w
        costs.loc[date] = c
        pnl.loc[date] = float(w @ next_ret) - c
        prev = w

    out = dict(base)
    out["base_net"] = base["net"]
    out["base_weights"] = base["weights"]
    out["net"] = pnl.dropna()
    out["costs"] = costs.reindex(out["net"].index).fillna(0.0)
    out["weights"] = weights.reindex(out["net"].index)
    out["risk_budgeted"] = True
    return out
