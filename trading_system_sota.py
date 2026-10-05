"""
Simons/Renaissance-inspired DAILY statistical-arbitrage research scaffold.

This is NOT Renaissance Technologies' proprietary system and cannot reproduce it.
It implements public quantitative principles associated with institutional stat-arb:
  * many weak, causal alphas across a liquid multi-asset universe;
  * cross-sectional prediction rather than one-asset up/down guessing;
  * walk-forward retraining with an embargo;
  * ensemble learning with simple models retained as controls;
  * regime-adaptive momentum/reversal blending;
  * market/beta-neutral portfolio construction;
  * Ledoit-Wolf shrinkage covariance;
  * cost-aware constrained optimization;
  * volatility targeting and position caps;
  * block-bootstrap Sharpe uncertainty and a deflated-Sharpe-style diagnostic;
  * synthetic self-tests for causality, null behavior, planted alpha, and constraints;
  * optional Yahoo data and Alpaca PAPER rebalance (dry-run by default).

Core equations
--------------
1) Cross-sectional target for asset i at date t

      y[i,t] = 1{ r[i,t+1] - median_j(r[j,t+1]) > 0 }

2) Machine-learning alpha from an ensemble probability p[i,t]

      a_ml[i,t] = log( p[i,t] / (1-p[i,t]) )

   then cross-sectionally z-scored so the alpha is relative rather than directional.

3) Residual momentum alpha

      r_resid[i,t,h] = r[i,t,h] - beta[i,t] * r_mkt[t,h]

4) Regime-aware alpha blend

      alpha = b_ml a_ml + b_mom(regime) z(r_resid_21)
                         + b_rev(regime) z(-r_5)
                         + b_vol z(-sigma_20)

5) Shrinkage risk model

      Sigma_LW = delta F + (1-delta) S

   where S is the sample covariance and F is a structured shrinkage target.

6) Cost-aware portfolio optimization

      w* = argmin_w [ -alpha' w
                       + lambda/2 * w' Sigma w
                       + kappa * sum_i c_i |w_i - w_prev_i| ]

      subject to
          1'w = 0                      (dollar neutral)
          beta'w = 0                   (market-beta neutral)
          ||w||_1 <= G                 (gross exposure cap)
          |w_i| <= w_max               (single-name cap)

7) Volatility targeting

      w <- w * min(1, sigma_target / sqrt(252 * w' Sigma w))

8) Daily PnL after costs

      R_p[t+1] = w_t' r_{t+1} - Cost(Delta w_t)

The implementation intentionally favors transparent mathematics, robust validation,
and deterministic behavior over adding fashionable models without evidence of edge.

Usage
-----
  pip install numpy pandas scipy scikit-learn yfinance alpaca-py
  python simons_style_stat_arb.py selftest
  python simons_style_stat_arb.py synthetic
  python trading_system_sota.py backtest --universe stocks
  python trading_system_sota.py backtest --universe crypto
  python trading_system_sota.py backtest --universe forex
  python trading_system_sota.py backtest --universe commodities
  python trading_system_sota.py live --universe stocks              # Alpaca PAPER, dry-run
  python trading_system_sota.py live --universe crypto --execute    # Alpaca PAPER crypto

Live execution matrix
---------------------
  stocks / ETFs       : supported by the bundled Alpaca PAPER adapter
  crypto              : supported by the bundled Alpaca PAPER adapter (fractional, GTC)
  spot forex          : research/backtest only until a forex broker adapter is configured
  commodity futures   : research/backtest only until a futures broker adapter is configured
  commodity ETFs      : supported through Alpaca as US equities/ETFs

Research software only; not investment advice.
"""
from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from dataclasses import dataclass, replace
from typing import Dict, Iterable, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import norm, skew, kurtosis
from sklearn.covariance import LedoitWolf
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

log = logging.getLogger("stat_arb")
TRADING_DAYS = 252
EPS = 1e-12

# Research universes. Yahoo symbols are used for historical data.
# Live Alpaca execution is intentionally narrower: US equities/ETFs and crypto only.
ASSET_UNIVERSES: Dict[str, Tuple[str, ...]] = {
    "etf_stat_arb": (
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "XLE", "XLV", "XLI", "XLP",
        "TLT", "GLD",
    ),
    "stocks": (
        "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "AVGO", "JPM",
        "XOM", "UNH", "COST", "HD",
    ),
    "crypto": ("BTC-USD", "ETH-USD", "SOL-USD", "XRP-USD", "ADA-USD", "AVAX-USD"),
    "forex": ("EURUSD=X", "GBPUSD=X", "USDJPY=X", "AUDUSD=X", "USDCAD=X", "USDCHF=X", "NZDUSD=X"),
    "commodities": ("GC=F", "SI=F", "CL=F", "NG=F", "HG=F", "ZC=F", "ZW=F", "ZS=F"),
    "commodity_etfs": ("GLD", "SLV", "USO", "UNG", "CPER", "DBA"),
}

# Yahoo -> Alpaca symbol normalization for crypto pairs.
ALPACA_CRYPTO_SYMBOLS: Dict[str, str] = {
    "BTC-USD": "BTC/USD", "ETH-USD": "ETH/USD", "SOL-USD": "SOL/USD",
    "XRP-USD": "XRP/USD", "ADA-USD": "ADA/USD", "AVAX-USD": "AVAX/USD",
}

def universe_tickers(name: str) -> Tuple[str, ...]:
    if name not in ASSET_UNIVERSES:
        raise ValueError(f"Unknown universe {name!r}; choose from {sorted(ASSET_UNIVERSES)}")
    return ASSET_UNIVERSES[name]

def infer_asset_class(symbol: str) -> str:
    if symbol in ALPACA_CRYPTO_SYMBOLS or symbol.endswith("-USD"):
        return "crypto"
    if symbol.endswith("=X"):
        return "forex"
    if symbol.endswith("=F"):
        return "commodity_future"
    return "equity"

def alpaca_symbol(symbol: str) -> str:
    return ALPACA_CRYPTO_SYMBOLS.get(symbol, symbol)


# --------------------------------------------------------------------------- config
@dataclass(frozen=True)
class Config:
    tickers: Tuple[str, ...] = ASSET_UNIVERSES["etf_stat_arb"]
    period: str = "12y"
    train_window: int = 756
    risk_window: int = 126
    beta_window: int = 126
    retrain_every: int = 21
    embargo: int = 1
    rf_trees: int = 160
    rf_depth: int = 5
    min_leaf: int = 40
    seed: int = 42

    gross_cap: float = 1.50
    name_cap: float = 0.20
    target_vol: float = 0.10
    risk_aversion: float = 10.0
    turnover_penalty: float = 0.50

    spread_bps: float = 1.0
    impact_coeff: float = 0.10
    min_adv_usd: float = 5_000_000.0

    alpha_ml: float = 0.60
    alpha_momentum: float = 0.25
    alpha_reversal: float = 0.12
    alpha_lowvol: float = 0.03

    bootstrap_samples: int = 500
    bootstrap_block: int = 10
    assumed_trials: int = 20


# --------------------------------------------------------------------------- helpers
def _zscore(s: pd.Series) -> pd.Series:
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd < EPS:
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / sd


def _winsorize_cs(frame: pd.DataFrame, q: float = 0.02) -> pd.DataFrame:
    def one(x: pd.Series) -> pd.Series:
        if x.notna().sum() < 4:
            return x
        lo, hi = x.quantile(q), x.quantile(1 - q)
        return x.clip(lo, hi)
    return frame.apply(one, axis=1)


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0.0)
    dn = -d.clip(upper=0.0)
    au = up.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = au / ad.replace(0.0, np.nan)
    out = 100.0 - 100.0 / (1.0 + rs)
    out = out.where(ad != 0.0, 100.0).where(au.notna())
    return out / 100.0


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = m.ewm(span=signal, adjust=False).mean()
    return (m - sig) / close


def bollinger_pb(close: pd.Series, n: int = 20) -> pd.Series:
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    return (close - (mid - 2 * sd)) / (4 * sd).replace(0.0, np.nan)


# --------------------------------------------------------------------------- data
def load_panel(tickers: Iterable[str], period: str) -> Dict[str, pd.DataFrame]:
    """Load real adjusted OHLCV. Never substitutes synthetic data on download failure."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("yfinance is required for real-data backtests: pip install yfinance") from e

    out: Dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = yf.download(ticker, period=period, interval="1d", auto_adjust=True, progress=False)
        if df is None or df.empty:
            raise RuntimeError(f"No data returned for {ticker}")
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        # Some Yahoo FX feeds do not provide economically meaningful volume. Keep price
        # rows and mark unavailable volume as NaN so volume-derived alphas become neutral
        # rather than fabricating liquidity.
        required = ["Open", "High", "Low", "Close"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise RuntimeError(f"Missing OHLC columns for {ticker}: {missing}")
        if "Volume" not in df.columns:
            df["Volume"] = np.nan
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df = df.dropna(subset=required)
        if (df["Volume"].fillna(0.0) <= 0).all():
            df["Volume"] = np.nan
        df.index = pd.to_datetime(df.index).tz_localize(None)
        out[ticker] = df
    return out


def align_panel(panel: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    idx = None
    for df in panel.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None or len(idx) == 0:
        raise RuntimeError("No common dates across assets")
    return {k: v.reindex(idx).ffill().dropna() for k, v in panel.items()}


# --------------------------------------------------------------------------- features
def market_return(close: pd.DataFrame) -> pd.Series:
    return np.log(close).diff().mean(axis=1)


def rolling_beta(ret: pd.Series, mkt: pd.Series, n: int) -> pd.Series:
    cov = ret.rolling(n).cov(mkt)
    var = mkt.rolling(n).var().replace(0.0, np.nan)
    return cov / var


def build_feature_cube(panel: Dict[str, pd.DataFrame], cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Return (features_long, close_wide, dollar_volume_wide).

    Every feature at t uses only information available at or before t.
    Cross-sectional transforms are performed inside each date only.
    """
    panel = align_panel(panel)
    close = pd.DataFrame({a: d["Close"] for a, d in panel.items()})
    high = pd.DataFrame({a: d["High"] for a, d in panel.items()})
    low = pd.DataFrame({a: d["Low"] for a, d in panel.items()})
    volume = pd.DataFrame({a: d["Volume"] for a, d in panel.items()})
    dollar_vol = close * volume

    logp = np.log(close)
    ret1 = logp.diff()
    mkt1 = market_return(close)
    mkt21 = mkt1.rolling(21).sum()

    frames = []
    for asset in close.columns:
        c = close[asset]
        r = ret1[asset]
        beta = rolling_beta(r, mkt1, cfg.beta_window)
        feat = pd.DataFrame(index=close.index)
        feat["ret_1"] = r
        feat["ret_5"] = logp[asset].diff(5)
        feat["ret_21"] = logp[asset].diff(21)
        feat["ret_63"] = logp[asset].diff(63)
        feat["resid_mom_21"] = feat["ret_21"] - beta * mkt21
        feat["vol_20"] = r.rolling(20).std() * math.sqrt(TRADING_DAYS)
        feat["vol_60"] = r.rolling(60).std() * math.sqrt(TRADING_DAYS)
        feat["down_vol_20"] = r.clip(upper=0).rolling(20).std() * math.sqrt(TRADING_DAYS)
        feat["rsi_14"] = rsi(c)
        feat["macd_hist"] = macd_hist(c)
        feat["boll_pb"] = bollinger_pb(c)
        feat["sma_20_100"] = c.rolling(20).mean() / c.rolling(100).mean() - 1.0
        feat["range_20"] = ((high[asset] - low[asset]) / c).rolling(20).mean()
        dv = dollar_vol[asset].replace(0.0, np.nan)
        feat["log_dollar_vol"] = np.log(dv).rolling(20).mean()
        feat["volume_shock"] = np.log(dv) - np.log(dv).rolling(20).mean()
        feat["beta_mkt"] = beta
        feat["asset"] = asset
        frames.append(feat)

    long = pd.concat(frames).set_index("asset", append=True)
    long.index.names = ["date", "asset"]

    # Cross-sectional robustification and z-scoring: each date independently.
    numerical = list(long.columns)
    zlong = pd.DataFrame(index=long.index, columns=numerical, dtype=float)
    for feature in numerical:
        mat = long[feature].unstack("asset")
        lo = mat.quantile(0.02, axis=1)
        hi = mat.quantile(0.98, axis=1)
        wins = mat.clip(lower=lo, upper=hi, axis=0)
        mu = wins.mean(axis=1)
        sd = wins.std(axis=1, ddof=0).replace(0.0, np.nan)
        z = wins.sub(mu, axis=0).div(sd, axis=0).fillna(0.0)
        zlong[feature] = z.stack(future_stack=True).reindex(long.index)

    # Keep cross-sectional versions for pooled ML; they are dimensionless and more stationary.
    return zlong.sort_index(), close, dollar_vol


def make_cross_sectional_target(close: pd.DataFrame) -> pd.Series:
    """y[i,t]=1 when next-day asset return exceeds the next-day cross-sectional median."""
    fwd = close.pct_change().shift(-1)
    excess = fwd.sub(fwd.median(axis=1), axis=0)
    y = (excess > 0).astype(float).where(fwd.notna())
    return y.stack().rename("target")


# --------------------------------------------------------------------------- regime
def infer_regime(close: pd.DataFrame, date: pd.Timestamp) -> str:
    """Causal 3-state regime using market trend and volatility."""
    hist = close.loc[:date]
    mkt = hist.pct_change().mean(axis=1)
    if len(mkt.dropna()) < 80:
        return "neutral"
    trend = (1 + mkt.tail(63)).prod() - 1
    rv20 = mkt.tail(20).std() * math.sqrt(TRADING_DAYS)
    rv_hist = mkt.rolling(20).std().dropna() * math.sqrt(TRADING_DAYS)
    high_vol = rv20 > rv_hist.tail(252).quantile(0.70) if len(rv_hist) else False
    if trend > 0.03 and not high_vol:
        return "trend"
    if trend < -0.03 or high_vol:
        return "stress"
    return "neutral"


def alpha_blend(proba: pd.Series, feat_today: pd.DataFrame, regime: str, cfg: Config) -> pd.Series:
    p = proba.clip(1e-4, 1 - 1e-4)
    ml = _zscore(np.log(p / (1 - p)))
    mom = _zscore(feat_today["resid_mom_21"])
    rev = _zscore(-feat_today["ret_5"])
    lowvol = _zscore(-feat_today["vol_20"])

    mom_w, rev_w = cfg.alpha_momentum, cfg.alpha_reversal
    if regime == "trend":
        mom_w *= 1.45
        rev_w *= 0.55
    elif regime == "stress":
        mom_w *= 0.65
        rev_w *= 1.35

    a = cfg.alpha_ml * ml + mom_w * mom + rev_w * rev + cfg.alpha_lowvol * lowvol
    return _zscore(a.replace([np.inf, -np.inf], np.nan).fillna(0.0))


# --------------------------------------------------------------------------- model
def fit_ensemble(x_train: pd.DataFrame, y_train: pd.Series, x_test: pd.DataFrame, cfg: Config) -> np.ndarray:
    keep = x_train.notna().all(axis=1) & y_train.notna()
    xtr, ytr = x_train.loc[keep], y_train.loc[keep].astype(int)
    xt = x_test.fillna(0.0)
    if len(xtr) < 100 or ytr.nunique() < 2:
        return np.full(len(xt), 0.5)

    lr = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.20, max_iter=1200, class_weight="balanced", random_state=cfg.seed),
    )
    rf = RandomForestClassifier(
        n_estimators=cfg.rf_trees,
        max_depth=cfg.rf_depth,
        min_samples_leaf=cfg.min_leaf,
        max_features="sqrt",
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=cfg.seed,
    )
    probs = []
    for model in (lr, rf):
        model.fit(xtr.values, ytr.values)
        probs.append(model.predict_proba(xt.values)[:, 1])
    return 0.55 * probs[0] + 0.45 * probs[1]


def walk_forward_probabilities(features: pd.DataFrame, target: pd.Series, cfg: Config) -> pd.Series:
    dates = features.index.get_level_values("date").unique().sort_values()
    out = pd.Series(np.nan, index=features.index, name="p")
    if len(dates) <= cfg.train_window + cfg.embargo:
        return out

    for t0 in range(cfg.train_window + cfg.embargo, len(dates), cfg.retrain_every):
        train_end = t0 - cfg.embargo
        train_dates = dates[max(0, train_end - cfg.train_window):train_end]
        test_dates = dates[t0:min(t0 + cfg.retrain_every, len(dates))]
        tr_idx = features.index.get_level_values("date").isin(train_dates)
        te_idx = features.index.get_level_values("date").isin(test_dates)
        xtr, ytr = features.loc[tr_idx], target.reindex(features.index[tr_idx])
        xte = features.loc[te_idx]
        out.loc[xte.index] = fit_ensemble(xtr, ytr, xte, cfg)
    return out


# --------------------------------------------------------------------------- portfolio math
def estimate_beta_vector(close_hist: pd.DataFrame, window: int) -> np.ndarray:
    r = np.log(close_hist).diff().dropna().tail(window)
    if len(r) < 20:
        return np.ones(close_hist.shape[1])
    m = r.mean(axis=1)
    var = m.var()
    if var < EPS:
        return np.ones(close_hist.shape[1])
    return np.array([r[c].cov(m) / var for c in r.columns], dtype=float)


def shrink_cov(close_hist: pd.DataFrame, window: int) -> np.ndarray:
    r = close_hist.pct_change().dropna().tail(window)
    if len(r) < max(20, r.shape[1] + 2):
        s = np.cov(r.values.T) if len(r) > 2 else np.eye(r.shape[1]) * 1e-4
        return np.atleast_2d(s)
    return LedoitWolf().fit(r.values).covariance_


def expected_cost_vector(dollar_vol_today: pd.Series, vol_diag: np.ndarray, cfg: Config) -> np.ndarray:
    """
    Linearized marginal trading-cost proxy.

    c_i ~= half-spread + eta * sigma_i / sqrt(ADV_i / ADV_ref)

    Used inside optimization; realized backtest cost uses a square-root turnover model.
    """
    adv = dollar_vol_today.clip(lower=cfg.min_adv_usd).values.astype(float)
    sigma = np.sqrt(np.maximum(vol_diag, EPS))
    liq = np.sqrt(cfg.min_adv_usd / adv)
    return cfg.spread_bps / 2e4 + cfg.impact_coeff * sigma * liq


def optimize_weights(alpha: pd.Series, cov: np.ndarray, beta: np.ndarray, prev: pd.Series,
                     dollar_vol_today: pd.Series, cfg: Config) -> pd.Series:
    assets = list(alpha.index)
    n = len(assets)
    a = alpha.values.astype(float)
    prevv = prev.reindex(assets).fillna(0.0).values.astype(float)
    beta = np.nan_to_num(beta, nan=1.0, posinf=1.0, neginf=1.0)
    cost = expected_cost_vector(dollar_vol_today.reindex(assets).fillna(cfg.min_adv_usd), np.diag(cov), cfg)

    def obj(w: np.ndarray) -> float:
        risk = 0.5 * cfg.risk_aversion * float(w @ cov @ w)
        turnover = cfg.turnover_penalty * float(np.sum(cost * np.sqrt((w - prevv) ** 2 + 1e-8)))
        return -float(a @ w) + risk + turnover

    cons = [
        {"type": "eq", "fun": lambda w: np.sum(w)},
        {"type": "eq", "fun": lambda w: float(beta @ w)},
        {"type": "ineq", "fun": lambda w: cfg.gross_cap - np.sum(np.abs(w))},
    ]
    bounds = [(-cfg.name_cap, cfg.name_cap)] * n
    x0 = prevv.copy()
    # Project initial point toward feasibility.
    x0 -= x0.mean()
    if np.sum(np.abs(x0)) > cfg.gross_cap:
        x0 *= cfg.gross_cap / np.sum(np.abs(x0))

    res = minimize(obj, x0=x0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"maxiter": 300, "ftol": 1e-9, "disp": False})
    if not res.success:
        # Deterministic fallback: alpha rank portfolio with beta projection.
        w = _zscore(alpha).values
        w = w - w.mean()
        bb = float(beta @ beta)
        if bb > EPS:
            w = w - beta * (beta @ w) / bb
        w = np.clip(w, -cfg.name_cap, cfg.name_cap)
        gross = np.sum(np.abs(w))
        if gross > EPS:
            w *= min(1.0, cfg.gross_cap / gross)
    else:
        w = res.x

    # Volatility target (never scales above existing gross exposure).
    ann_vol = math.sqrt(max(TRADING_DAYS * float(w @ cov @ w), 0.0))
    if ann_vol > cfg.target_vol > 0:
        w *= cfg.target_vol / ann_vol
    return pd.Series(w, index=assets)


def realized_cost(delta_w: pd.Series, dollar_vol_today: pd.Series, ret_vol: pd.Series, cfg: Config) -> float:
    """Square-root impact proxy in portfolio-return units."""
    dw = delta_w.abs()
    spread = (cfg.spread_bps / 1e4) * dw.sum()
    adv = dollar_vol_today.reindex(dw.index).clip(lower=cfg.min_adv_usd)
    sigma = ret_vol.reindex(dw.index).fillna(ret_vol.median() if ret_vol.notna().any() else 0.02)
    impact = cfg.impact_coeff * np.sum(dw * sigma * np.sqrt(np.maximum(dw, 0) * cfg.min_adv_usd / adv))
    return float(spread + impact)


# --------------------------------------------------------------------------- backtest
def run_backtest(panel: Dict[str, pd.DataFrame], cfg: Config) -> dict:
    panel = align_panel(panel)
    features, close, dollar_vol = build_feature_cube(panel, cfg)
    target = make_cross_sectional_target(close).reindex(features.index)
    prob = walk_forward_probabilities(features, target, cfg)

    dates = prob.dropna().index.get_level_values("date").unique().sort_values()
    if len(dates) < 20:
        raise RuntimeError("Not enough out-of-sample predictions")

    weights = pd.DataFrame(0.0, index=dates, columns=close.columns)
    pnl = pd.Series(index=dates, dtype=float)
    costs = pd.Series(index=dates, dtype=float)
    regimes = {}
    prev = pd.Series(0.0, index=close.columns)
    ret = close.pct_change()

    for k, date in enumerate(dates):
        if date not in close.index or close.index.get_loc(date) >= len(close.index) - 1:
            continue
        p = prob.xs(date, level="date").reindex(close.columns).fillna(0.5)
        feat_today = features.xs(date, level="date").reindex(close.columns).fillna(0.0)
        regime = infer_regime(close, date)
        regimes[date] = regime
        alpha = alpha_blend(p, feat_today, regime, cfg).reindex(close.columns).fillna(0.0)

        hist = close.loc[:date]
        cov = shrink_cov(hist, cfg.risk_window)
        beta = estimate_beta_vector(hist, cfg.beta_window)
        adv = dollar_vol.loc[date].reindex(close.columns).fillna(cfg.min_adv_usd)
        w = optimize_weights(alpha, cov, beta, prev, adv, cfg)

        next_date = close.index[close.index.get_loc(date) + 1]
        next_ret = ret.loc[next_date].reindex(close.columns).fillna(0.0)
        vol20 = ret.loc[:date].tail(20).std() * math.sqrt(TRADING_DAYS)
        c = realized_cost(w - prev, adv, vol20, cfg)
        pnl.loc[date] = float(w @ next_ret) - c
        costs.loc[date] = c
        weights.loc[date] = w
        prev = w

    pnl = pnl.dropna()
    costs = costs.reindex(pnl.index).fillna(0.0)
    weights = weights.reindex(pnl.index)

    # Baselines on the same OOS dates.
    next_ret_matrix = pd.DataFrame(index=pnl.index, columns=close.columns, dtype=float)
    for date in pnl.index:
        nxt = close.index[close.index.get_loc(date) + 1]
        next_ret_matrix.loc[date] = ret.loc[nxt]

    ew = next_ret_matrix.mean(axis=1)
    # Simple cross-sectional 21d momentum long-short baseline.
    mom_pnl = []
    for date in pnl.index:
        score = np.log(close).diff(21).loc[date]
        z = _zscore(score.fillna(0.0))
        w = z / max(z.abs().sum(), EPS) * cfg.gross_cap
        mom_pnl.append(float(w @ next_ret_matrix.loc[date]))
    mom = pd.Series(mom_pnl, index=pnl.index)

    return {
        "net": pnl,
        "costs": costs,
        "weights": weights,
        "prob": prob,
        "target": target,
        "equal_weight": ew,
        "momentum_ls": mom,
        "regimes": pd.Series(regimes),
    }


# --------------------------------------------------------------------------- statistics
def performance_metrics(r: pd.Series) -> dict:
    r = pd.Series(r).dropna()
    n = len(r)
    if n == 0:
        return {"cagr": np.nan, "sharpe": np.nan, "max_dd": np.nan, "vol": np.nan, "days": 0}
    eq = (1 + r).cumprod()
    years = n / TRADING_DAYS
    sd = r.std(ddof=1)
    sh = r.mean() / sd * math.sqrt(TRADING_DAYS) if sd > EPS else 0.0
    return {
        "cagr": eq.iloc[-1] ** (1 / years) - 1 if eq.iloc[-1] > 0 else -1.0,
        "sharpe": sh,
        "max_dd": (eq / eq.cummax() - 1).min(),
        "vol": sd * math.sqrt(TRADING_DAYS),
        "days": n,
    }


def block_bootstrap_sharpe_ci(r: pd.Series, samples: int = 500, block: int = 10,
                              seed: int = 42) -> Tuple[float, float]:
    x = pd.Series(r).dropna().values
    n = len(x)
    if n < 40:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(samples):
        out = []
        while len(out) < n:
            start = int(rng.integers(0, max(1, n - block + 1)))
            out.extend(x[start:start + block])
        z = np.asarray(out[:n])
        sd = z.std(ddof=1)
        vals.append(z.mean() / sd * math.sqrt(TRADING_DAYS) if sd > EPS else 0.0)
    return float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))


def probabilistic_sharpe_ratio(r: pd.Series, benchmark_sharpe: float = 0.0) -> float:
    """Bailey/Lopez-de-Prado-style PSR approximation accounting for skew/kurtosis."""
    x = pd.Series(r).dropna().values
    n = len(x)
    if n < 3 or np.std(x, ddof=1) < EPS:
        return 0.5
    sr = np.mean(x) / np.std(x, ddof=1) * math.sqrt(TRADING_DAYS)
    g3 = float(skew(x, bias=False))
    g4 = float(kurtosis(x, fisher=False, bias=False))
    # Use per-period Sharpe inside sampling formula.
    sr_d = sr / math.sqrt(TRADING_DAYS)
    sr0_d = benchmark_sharpe / math.sqrt(TRADING_DAYS)
    denom = math.sqrt(max(1 - g3 * sr_d + ((g4 - 1) / 4) * sr_d * sr_d, EPS))
    z = (sr_d - sr0_d) * math.sqrt(n - 1) / denom
    return float(norm.cdf(z))


def deflated_sharpe_probability(r: pd.Series, n_trials: int = 20) -> float:
    """
    Conservative DSR-style probability.

    We estimate the expected maximum Sharpe from n_trials under the null and feed that
    as the benchmark into PSR. This is an approximation, explicitly reported as such.
    """
    x = pd.Series(r).dropna()
    n = len(x)
    if n < 3:
        return 0.5
    # Expected max of M standard normal draws, mapped to annualized Sharpe sampling scale.
    m = max(int(n_trials), 1)
    q = norm.ppf((m - 0.375) / (m + 0.25)) if m > 1 else 0.0
    sr_noise = q * math.sqrt(TRADING_DAYS / max(n - 1, 1))
    return probabilistic_sharpe_ratio(x, benchmark_sharpe=sr_noise)


def directional_ic(prob: pd.Series, target: pd.Series) -> float:
    df = pd.concat([prob.rename("p"), target.rename("y")], axis=1).dropna()
    if len(df) < 10:
        return float("nan")
    # Cross-sectional rank correlation per date, then average.
    vals = []
    for _, g in df.groupby(level="date"):
        if g["p"].nunique() > 1 and g["y"].nunique() > 1:
            vals.append(g["p"].corr(g["y"], method="spearman"))
    return float(np.nanmean(vals)) if vals else float("nan")


def print_report(out: dict, cfg: Config) -> None:
    print("\n=== Simons-style cross-sectional walk-forward report ===")
    for name, r in [("model", out["net"]), ("equal_weight", out["equal_weight"]), ("momentum_ls", out["momentum_ls"])]:
        m = performance_metrics(r)
        lo, hi = block_bootstrap_sharpe_ci(r, cfg.bootstrap_samples, cfg.bootstrap_block, cfg.seed)
        print(f"{name:>13}: CAGR {m['cagr']:+7.2%} | Sharpe {m['sharpe']:+5.2f} "
              f"| boot95 [{lo:+.2f},{hi:+.2f}] | Vol {m['vol']:.2%} | MaxDD {m['max_dd']:.2%}")
    ic = directional_ic(out["prob"], out["target"])
    psr = probabilistic_sharpe_ratio(out["net"], 0.0)
    dsr = deflated_sharpe_probability(out["net"], cfg.assumed_trials)
    avg_gross = out["weights"].abs().sum(axis=1).mean()
    avg_net = out["weights"].sum(axis=1).abs().mean()
    avg_turn = out["weights"].diff().abs().sum(axis=1).mean() * TRADING_DAYS
    print(f"IC(rank): {ic:+.4f} | PSR(SR>0): {psr:.3f} | approximate DSR probability: {dsr:.3f}")
    print(f"Average gross: {avg_gross:.3f} | average |net|: {avg_net:.6f} | annualized turnover: {avg_turn:.1f}x")
    print(f"Mean modeled trading cost/day: {out['costs'].mean():.6%}")


# --------------------------------------------------------------------------- synthetic data and tests
def synthetic_panel(n: int = 1200, assets: int = 10, seed: int = 7, planted: bool = True) -> Dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    names = [f"A{i:02d}" for i in range(assets)]
    idx = pd.bdate_range("2014-01-01", periods=n)
    market = rng.normal(0, 0.007, n)
    rets = np.zeros((n, assets))
    chars = rng.normal(size=assets)
    for t in range(1, n):
        eps = rng.normal(0, 0.009, assets)
        signal = np.zeros(assets)
        if planted and t >= 6:
            # weak relative mean-reversion + persistent asset characteristic; cross-sectionally centered
            prev5 = rets[max(0, t-5):t].sum(axis=0)
            signal = -0.06 * (prev5 - prev5.mean()) + 0.0004 * (chars - chars.mean())
        rets[t] = 0.55 * market[t] + eps + signal
    prices = 100 * np.exp(np.cumsum(rets, axis=0))
    out = {}
    for j, a in enumerate(names):
        close = prices[:, j]
        intr = np.maximum(0.001, np.abs(rng.normal(0.008, 0.003, n)))
        op = close * np.exp(rng.normal(0, 0.0015, n))
        hi = np.maximum(op, close) * (1 + intr / 2)
        lo = np.minimum(op, close) * (1 - intr / 2)
        vol = rng.lognormal(mean=15.5, sigma=0.3, size=n)
        out[a] = pd.DataFrame({"Open": op, "High": hi, "Low": lo, "Close": close, "Volume": vol}, index=idx)
    return out


def selftest() -> int:
    failures = []
    def check(name: str, ok: bool, detail: str = ""):
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} {detail}")
        if not ok:
            failures.append(name)

    cfg = replace(Config(), tickers=tuple(f"A{i:02d}" for i in range(6)), train_window=180,
                  risk_window=60, beta_window=60, retrain_every=30, rf_trees=30,
                  min_leaf=12, bootstrap_samples=60)

    print("1. Causality")
    p = synthetic_panel(420, 6, seed=1, planted=False)
    f1, _, _ = build_feature_cube(p, cfg)
    p2 = {k: v.copy() for k, v in p.items()}
    cutoff = list(p2.values())[0].index[320]
    for df in p2.values():
        future = df.index > cutoff
        df.loc[future, "Close"] *= np.linspace(0.7, 1.3, future.sum())
    f2, _, _ = build_feature_cube(p2, cfg)
    common = f1.index[f1.index.get_level_values("date") <= cutoff]
    same = np.allclose(f1.loc[common].values, f2.loc[common].values, equal_nan=True)
    check("future price edits do not change past features", same)

    print("2. Indicator sanity")
    up = pd.Series(np.arange(1.0, 80.0))
    check("RSI monotonic rise = 1", abs(rsi(up).iloc[-1] - 1.0) < 1e-9)
    check("RSI monotonic fall = 0", abs(rsi(up[::-1].reset_index(drop=True)).iloc[-1]) < 1e-9)

    print("3. Null synthetic market")
    null = run_backtest(synthetic_panel(460, 6, seed=22, planted=False), cfg)
    ic0 = directional_ic(null["prob"], null["target"])
    check("null IC remains small", abs(ic0) < 0.12, f"(IC={ic0:+.3f})")

    print("4. Planted weak cross-sectional alpha")
    planted = run_backtest(synthetic_panel(460, 6, seed=23, planted=True), cfg)
    ic1 = directional_ic(planted["prob"], planted["target"])
    check("planted alpha produces positive IC", ic1 > 0.02, f"(IC={ic1:+.3f})")

    print("5. Portfolio constraints")
    w = planted["weights"].dropna()
    check("dollar neutrality", float(w.sum(axis=1).abs().max()) < 2e-3,
          f"(max |net|={w.sum(axis=1).abs().max():.5f})")
    check("gross cap", float(w.abs().sum(axis=1).max()) <= cfg.gross_cap + 1e-6,
          f"(max gross={w.abs().sum(axis=1).max():.3f})")
    check("single-name cap", float(w.abs().max().max()) <= cfg.name_cap + 1e-6,
          f"(max weight={w.abs().max().max():.3f})")

    print("6. Statistics")
    lo, hi = block_bootstrap_sharpe_ci(planted["net"], 100, 8, 5)
    check("bootstrap interval finite", np.isfinite(lo) and np.isfinite(hi) and lo <= hi,
          f"([{lo:+.2f},{hi:+.2f}])")
    dsr = deflated_sharpe_probability(planted["net"], 10)
    check("DSR-style probability valid", 0 <= dsr <= 1, f"({dsr:.3f})")

    print("\nALL PASSED" if not failures else f"\nFAILED: {failures}")
    return 1 if failures else 0


# --------------------------------------------------------------------------- Alpaca PAPER live rebalance
class PaperRebalancer:
    """One rebalance from the latest target weights. Alpaca PAPER is hard-wired."""
    def __init__(self, cfg: Config, execute: bool = False):
        try:
            from alpaca.trading.client import TradingClient
        except ImportError as e:
            raise RuntimeError("alpaca-py is required for live PAPER mode: pip install alpaca-py") from e
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_SECRET_KEY")
        if not key or not secret:
            raise RuntimeError("Set ALPACA_API_KEY and ALPACA_SECRET_KEY")
        self.client = TradingClient(key, secret, paper=True)
        self.cfg = cfg
        self.execute = execute

    def rebalance(self, weights: pd.Series, prices: pd.Series) -> list:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest

        unsupported = [s for s in weights.index if infer_asset_class(s) in {"forex", "commodity_future"}]
        if unsupported:
            raise RuntimeError(
                "Alpaca PAPER adapter in this file does not execute spot FX or commodity futures: "
                + ", ".join(unsupported)
                + ". Backtest them here, or add a broker adapter (e.g. IBKR/OANDA/futures broker). "
                  "For commodity exposure on Alpaca use the commodity_etfs universe."
            )

        equity = float(self.client.get_account().equity)
        held_raw = {p.symbol: float(p.qty) for p in self.client.get_all_positions()}
        held = {}
        for k, v in held_raw.items():
            held[k] = v
            held[k.replace("/", "")] = v

        orders = []
        for symbol, w in weights.items():
            if symbol not in prices or prices[symbol] <= 0:
                continue
            cls = infer_asset_class(symbol)
            broker_symbol = alpaca_symbol(symbol)
            target_notional = equity * float(w)
            current_qty = float(held.get(broker_symbol, held.get(broker_symbol.replace("/", ""), 0.0)))
            current_notional = current_qty * float(prices[symbol])
            delta_notional = target_notional - current_notional

            # Avoid tiny churn. This is not a broker minimum; it is a research/live guardrail.
            if abs(delta_notional) < max(1.0, 0.0001 * equity):
                continue

            side = OrderSide.BUY if delta_notional > 0 else OrderSide.SELL
            if cls == "crypto":
                # Alpaca crypto is fractional and trades 24/7; GTC is supported.
                qty = abs(delta_notional) / float(prices[symbol])
                req = MarketOrderRequest(symbol=broker_symbol, qty=round(qty, 8),
                                         side=side, time_in_force=TimeInForce.GTC)
            else:
                qty = math.floor(abs(delta_notional) / float(prices[symbol]))
                if qty < 1:
                    continue
                req = MarketOrderRequest(symbol=broker_symbol, qty=qty,
                                         side=side, time_in_force=TimeInForce.DAY)

            info = {"symbol": symbol, "broker_symbol": broker_symbol, "asset_class": cls,
                    "target_weight": float(w), "delta_notional": float(delta_notional),
                    "submitted": False}
            if self.execute:
                self.client.submit_order(order_data=req)
                info["submitted"] = True
            orders.append(info)
        return orders



def latest_weights(panel: Dict[str, pd.DataFrame], cfg: Config) -> pd.Series:
    features, close, dv = build_feature_cube(panel, cfg)
    target = make_cross_sectional_target(close).reindex(features.index)
    dates = features.index.get_level_values("date").unique().sort_values()
    if len(dates) < cfg.train_window + cfg.embargo + 1:
        raise RuntimeError("Insufficient history")
    date = dates[-1]
    train_dates = dates[-(cfg.train_window + cfg.embargo):-cfg.embargo]
    tr = features.index.get_level_values("date").isin(train_dates)
    te = features.index.get_level_values("date") == date
    p = pd.Series(fit_ensemble(features.loc[tr], target.loc[features.index[tr]], features.loc[te], cfg),
                  index=features.loc[te].index).droplevel("date")
    feat_today = features.xs(date, level="date").reindex(close.columns).fillna(0.0)
    alpha = alpha_blend(p.reindex(close.columns).fillna(0.5), feat_today, infer_regime(close, date), cfg)
    cov = shrink_cov(close.loc[:date], cfg.risk_window)
    beta = estimate_beta_vector(close.loc[:date], cfg.beta_window)
    return optimize_weights(alpha, cov, beta, pd.Series(0.0, index=close.columns), dv.loc[date], cfg)


# --------------------------------------------------------------------------- CLI
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["selftest", "synthetic", "backtest", "live"])
    ap.add_argument("--period", default="12y")
    ap.add_argument("--universe", choices=sorted(ASSET_UNIVERSES), default="etf_stat_arb",
                    help="predefined research universe; ignored when --tickers is supplied")
    ap.add_argument("--tickers", nargs="*", default=None,
                    help="Yahoo symbols, e.g. AAPL MSFT, BTC-USD ETH-USD, EURUSD=X, GC=F")
    ap.add_argument("--execute", action="store_true", help="submit orders to Alpaca PAPER account")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    chosen = tuple(args.tickers) if args.tickers else universe_tickers(args.universe)
    cfg = Config(period=args.period, tickers=chosen)

    if args.command == "selftest":
        return selftest()
    if args.command == "synthetic":
        scfg = replace(cfg, tickers=tuple(f"A{i:02d}" for i in range(10)), train_window=300,
                       risk_window=100, beta_window=100, retrain_every=20, rf_trees=80, min_leaf=20)
        out = run_backtest(synthetic_panel(900, 10, seed=17, planted=True), scfg)
        print_report(out, scfg)
        return 0

    panel = load_panel(cfg.tickers, cfg.period)
    if args.command == "backtest":
        out = run_backtest(panel, cfg)
        print_report(out, cfg)
        return 0

    w = latest_weights(panel, cfg)
    prices = pd.Series({a: panel[a]["Close"].iloc[-1] for a in cfg.tickers})
    print("Latest target weights:")
    print(w.sort_values().to_string())
    orders = PaperRebalancer(cfg, execute=args.execute).rebalance(w, prices)
    for o in orders:
        print(o)
    return 0


if __name__ == "__main__":
    sys.exit(main())