"""Lightweight AI diagnostics for trading research: anomaly detection, confidence and explanations."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.metrics import mean_squared_error


@dataclass
class AIInsight:
    anomaly_score: float
    anomaly: bool
    confidence: float
    regime: str
    explanation: str


class MarketAICopilot:
    def __init__(self, seed: int = 42):
        self.seed = seed
        self.anomaly_model = IsolationForest(n_estimators=200, contamination=0.03, random_state=seed)

    def anomaly_scan(self, features: pd.DataFrame) -> pd.DataFrame:
        x = features.replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(x) < 30:
            return pd.DataFrame({"anomaly_score": np.zeros(len(x)), "anomaly": False}, index=x.index)
        pred = self.anomaly_model.fit_predict(x.values)
        score = -self.anomaly_model.score_samples(x.values)
        return pd.DataFrame({"anomaly_score": score, "anomaly": pred < 0}, index=x.index)

    @staticmethod
    def confidence_from_probabilities(p: pd.Series, dispersion: pd.Series | None = None) -> pd.Series:
        base = (p - 0.5).abs() * 2.0
        if dispersion is not None:
            base *= (1.0 - dispersion.clip(0, 1))
        return base.clip(0, 1)

    @staticmethod
    def explain_signal(symbol: str, alpha: float, features: pd.Series, regime: str,
                       anomaly: bool = False, confidence: float = 0.5) -> str:
        ranked = features.abs().sort_values(ascending=False).head(4)
        drivers = []
        for k in ranked.index:
            v = float(features[k])
            direction = "positive" if v > 0 else "negative"
            drivers.append(f"{k}={v:+.2f} ({direction})")
        stance = "long" if alpha > 0 else "short" if alpha < 0 else "flat"
        warning = " Anomaly filter is active, so confidence should be discounted." if anomaly else ""
        return (f"{symbol}: {stance} alpha {alpha:+.3f}, confidence {confidence:.0%}, regime={regime}. "
                f"Largest standardized drivers: {', '.join(drivers)}.{warning}")

    def forecast_risk(self, returns: pd.DataFrame, horizon: int = 1) -> Dict[str, dict]:
        """Small nonlinear model for next-period absolute return; diagnostic, not execution alpha."""
        out = {}
        for c in returns.columns:
            r = returns[c].dropna()
            if len(r) < 120:
                continue
            df = pd.DataFrame({
                "abs1": r.abs(), "abs5": r.abs().rolling(5).mean(),
                "vol20": r.rolling(20).std(), "ret1": r,
            }).dropna()
            y = r.abs().shift(-horizon).reindex(df.index)
            good = y.notna()
            X, y = df.loc[good], y.loc[good]
            split = int(len(X) * 0.8)
            if split < 50 or len(X) - split < 10:
                continue
            m = RandomForestRegressor(n_estimators=180, max_depth=5, min_samples_leaf=12,
                                      random_state=self.seed, n_jobs=-1)
            m.fit(X.iloc[:split], y.iloc[:split])
            pred = m.predict(X.iloc[split:])
            rmse = float(mean_squared_error(y.iloc[split:], pred) ** 0.5)
            nxt = float(m.predict(df.iloc[[-1]])[0])
            out[c] = {"predicted_abs_return": nxt, "oos_rmse": rmse}
        return out


def copilot_answer(question: str, *, metrics: dict, weights: pd.Series,
                   features_today: pd.DataFrame, regime: str,
                   risk_forecast: Dict[str, dict] | None = None) -> str:
    """Grounded local copilot for the dashboard; answers only from loaded research state."""
    q = (question or "").strip().lower()
    risk_forecast = risk_forecast or {}
    top_long = weights.sort_values(ascending=False).head(3)
    top_short = weights.sort_values().head(3)
    gross = float(weights.abs().sum())
    net = float(weights.sum())

    if any(k in q for k in ("risk", "danger", "drawdown", "volatile", "volatility")):
        risky = sorted(
            ((s, d.get("predicted_abs_return")) for s, d in risk_forecast.items()
             if d.get("predicted_abs_return") is not None),
            key=lambda x: x[1], reverse=True
        )[:5]
        tail = ", ".join(f"{s} {v:.2%}" for s, v in risky) if risky else "no forecast available"
        return (
            f"Current research regime is {regime}. Portfolio gross exposure is {gross:.2f}x and "
            f"net exposure is {net:+.3f}. Backtest max drawdown is {metrics.get('max_dd', float('nan')):.1%}, "
            f"with annualized volatility {metrics.get('vol', float('nan')):.1%}. "
            f"Highest predicted next-period absolute moves: {tail}."
        )

    if any(k in q for k in ("long", "buy", "bull")):
        txt = ", ".join(f"{s} {w:+.2%}" for s, w in top_long.items())
        return f"Largest positive target weights are {txt}. These are model targets, not discretionary buy recommendations."

    if any(k in q for k in ("short", "sell", "bear")):
        txt = ", ".join(f"{s} {w:+.2%}" for s, w in top_short.items())
        return f"Largest negative target weights are {txt}. These are model targets inside a hedged portfolio."

    if any(k in q for k in ("why", "driver", "explain", "signal")):
        symbol = next((s for s in weights.index if s.lower() in q), weights.abs().idxmax())
        row = features_today.loc[symbol].abs().sort_values(ascending=False).head(5)
        drivers = ", ".join(f"{k}={features_today.loc[symbol, k]:+.2f}" for k in row.index)
        return (
            f"{symbol} has target weight {weights.get(symbol, 0.0):+.2%} in a {regime} regime. "
            f"Its strongest standardized inputs today are {drivers}. "
            "The production signal remains the statistical ensemble and optimizer; this explanation layer does not override it."
        )

    return (
        f"Desk summary: regime={regime}; Sharpe={metrics.get('sharpe', float('nan')):.2f}; "
        f"CAGR={metrics.get('cagr', float('nan')):.1%}; volatility={metrics.get('vol', float('nan')):.1%}; "
        f"max drawdown={metrics.get('max_dd', float('nan')):.1%}; gross={gross:.2f}x; net={net:+.3f}. "
        "Ask about risk, the largest longs/shorts, or why a specific symbol has its current weight."
    )
