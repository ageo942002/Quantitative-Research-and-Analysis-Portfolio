# Final_project_functions_extended.py  

from typing import Optional, Tuple, Dict, List
import warnings, sys, time, json, random
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

from sklearn import __version__ as skl_version
from sklearn.base import clone
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import average_precision_score
import pandas_ta_classic as ta


import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import AUC, Precision, Recall
from tensorflow.keras.utils import timeseries_dataset_from_array
from sklearn.preprocessing import StandardScaler
import os, datetime as dt
import quantstats as qs
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Bidirectional
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.utils import timeseries_dataset_from_array
import os, random, tensorflow as tf, keras_tuner as kt
from tensorflow.keras import layers, models, callbacks
from sklearn.metrics import f1_score
from sklearn.base import clone

# helpers for reproductibilty

def set_seeds(seed: int = 42):
    np.random.seed(seed)
    random.seed(seed)

# Data download and resampling helpers

def _tz_naive_index(df):
    if isinstance(df.index, pd.DatetimeIndex) and getattr(df.index, "tz", None) is not None:
        df.index = df.index.tz_localize(None)
    return df

def _ensure_ohlcv_cols(df):
    keep = [c for c in ["Open","High","Low","Close","Adj Close","Volume"] if c in df.columns]
    return df[keep]

def get_daily(ticker, start, end=None, retries=3, wait=1.0):
    tk = yf.Ticker(ticker); last_err = None
    for i in range(retries):
        try:
            df = tk.history(start=start, end=end, interval="1d", auto_adjust=False, actions=False)
            df = _tz_naive_index(df).sort_index()
            if df.empty: raise RuntimeError("Empty frame from Yahoo")
            df = _ensure_ohlcv_cols(df)
            if "Adj Close" not in df.columns and "Close" in df.columns:
                df["Adj Close"] = df["Close"]
            return df.dropna(how="any")
        except Exception as e:
            last_err = e; time.sleep(wait * (2**i))
    raise RuntimeError(f"yfinance daily download failed for {ticker}: {last_err}")

def to_weekly(df_daily, rule="W-WED"):
    agg = {"Open":"first","High":"max","Low":"min","Close":"last","Adj Close":"last"}
    if "Volume" in df_daily.columns: agg["Volume"] = "sum"
    return df_daily.resample(rule).agg(agg).dropna(how="any")

# Core series and returns helpers log transforation

def to_1d_series(x) -> pd.Series:
    if isinstance(x, pd.DataFrame):
        s = x.iloc[:, 0] if x.shape[1] >= 1 else pd.Series(dtype=float)
    elif isinstance(x, pd.Series):
        s = x
    else:
        s = pd.Series(x)
    return pd.Series(s.values, index=pd.DatetimeIndex(s.index)).sort_index()

def as_return_series(x, price_col="Close", ret_col="Return", assume_returns="auto") -> pd.Series:
    if isinstance(x, pd.DataFrame):
        if ret_col in x.columns: s, is_ret = x[ret_col].copy(), True
        elif price_col in x.columns: s, is_ret = x[price_col].copy(), False
        elif x.shape[1] == 1: s, is_ret = x.iloc[:, 0].copy(), None
        else: raise ValueError("Provide Series or a DataFrame with Close/Return (or single column).")
    elif isinstance(x, pd.Series):
        s, is_ret = x.copy(), None
    else:
        s, is_ret = pd.Series(x), None
    try: s.index = pd.to_datetime(s.index).tz_localize(None)
    except Exception: pass
    s = s.astype(float)
    if is_ret is None:
        if assume_returns is True: is_ret = True
        elif assume_returns is False: is_ret = False
        else: is_ret = (np.nanmedian(np.abs(s.values)) < 0.5)
    r = s if is_ret else np.log(s).diff()
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna().sort_index()
    if r.index.has_duplicates: r = r[~r.index.duplicated(keep="last")]
    return r

# Weekly to Daily mapping

def to_daily_filled(w_series: pd.Series, daily_index: pd.DatetimeIndex) -> pd.Series:
    w = pd.Series(w_series).sort_index()
    d_idx = pd.DatetimeIndex(daily_index)
    tmp = pd.Series(index=d_idx.union(w.index), dtype=float)
    tmp.loc[w.index] = w.values
    out = tmp.reindex(d_idx).ffill()
    out.loc[: w.index.min()] = np.nan
    return out.dropna()

# Backtest & metrics using log returns

def backtest_from_positions(ret_daily: pd.Series,
                            pos_daily: pd.Series,
                            tx_bps: float = 5,
                            regime: Optional[pd.Series] = None) -> pd.Series:
    r = to_1d_series(ret_daily).astype(float)
    p = pd.Series(pos_daily).reindex(r.index).fillna(0.0).clip(0, 1)
    if regime is not None:
        g = pd.Series(regime).reindex(r.index).ffill().fillna(1.0).clip(0, 1)
        p = p * g
    turn = p.diff().abs().fillna(0.0)
    cost = turn * (tx_bps / 1e4)
    return (p * r - cost).replace([np.inf, -np.inf], np.nan).dropna()

def equity_and_perf(logret: pd.Series, start_idx=None) -> Tuple[pd.Series, pd.Series, Dict[str, float]]:
    r = to_1d_series(logret).dropna()
    if start_idx is not None: r = r.loc[r.index >= pd.to_datetime(start_idx)]
    if r.empty:
        return pd.Series(dtype=float), pd.Series(dtype=float), {"CAGR":np.nan,"Sharpe":np.nan,
               "Sortino":np.nan,"MaxDD":np.nan,"Calmar":np.nan,"WinRate":np.nan}
    eq = np.exp(r.cumsum())
    dd = eq/eq.cummax() - 1
    ann = np.exp(r.mean()*252) - 1
    vol = r.std(ddof=1)
    sharpe = (r.mean()/vol)*np.sqrt(252) if vol>0 else np.nan
    dn = r[r<0].std(ddof=1)
    sortino = (r.mean()/dn)*np.sqrt(252) if dn>0 else np.nan
    calmar = ann/abs(dd.min()) if dd.min()<0 else np.nan
    return eq, dd, {"CAGR":float(ann), "Sharpe":float(sharpe),
                    "Sortino":float(sortino), "MaxDD":float(dd.min()),
                    "Calmar":float(calmar), "WinRate":float((r>0).mean())}

# Baselines 

def ts_mom_signal_w(close_w: pd.Series, lookback=52, skip=1) -> pd.Series:
    m = np.log(close_w).shift(skip) - np.log(close_w).shift(lookback+skip)
    return (m > 0).astype(float).shift(1).dropna()

def sma_signal_w(close_w: pd.Series, s1=10, s2=40) -> pd.Series:
    s1v = close_w.rolling(s1).mean(); s2v = close_w.rolling(s2).mean()
    return (s1v > s2v).astype(float).shift(1).dropna()

# Model helprs

def to_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    d = model.decision_function(X)
    return (d - d.min())/(d.max() - d.min() + 1e-12)

def time_series_oof_proba(estimator, X_df: pd.DataFrame, y_s: pd.Series, cv) -> Tuple[pd.Series, pd.Series]:
    oof = pd.Series(index=X_df.index, dtype=float)
    for tr_idx, te_idx in cv.split(X_df, y_s):
        m = clone(estimator).fit(X_df.iloc[tr_idx], y_s.iloc[tr_idx])
        oof.iloc[te_idx] = to_proba(m, X_df.iloc[te_idx])
    oof = oof.dropna()
    return oof, y_s.loc[oof.index]

def f1_at_threshold(p: np.ndarray, y: np.ndarray, t: float) -> Tuple[float, float, float]:
    yhat = (p >= t).astype(int)
    tp = ((yhat==1)&(y==1)).sum(); fp = ((yhat==1)&(y==0)).sum(); fn = ((yhat==0)&(y==1)).sum()
    prec = tp/(tp+fp+1e-12); rec = tp/(tp+fn+1e-12)
    f1 = 2*prec*rec/(prec+rec+1e-12)
    return float(prec), float(rec), float(f1)

def select_thresholds(oof_proba_s: pd.Series, y_oof_s: pd.Series,
                      thr_grid: Optional[np.ndarray]=None,
                      ret_daily: Optional[pd.Series]=None,
                      train_end: Optional[pd.Timestamp]=None) -> Dict[str,float]:
    if thr_grid is None: thr_grid = np.linspace(0.05, 0.95, 37)
    p = oof_proba_s.values; y = y_oof_s.values
    f1_vals = [f1_at_threshold(p,y,t)[2] for t in thr_grid]
    thr_f1 = float(thr_grid[int(np.argmax(f1_vals))])
    thr_sharpe = None
    if (ret_daily is not None) and (train_end is not None):
        ret_d = to_1d_series(ret_daily)
        sharpe_vals = []
        for t in thr_grid:
            sig_w = pd.Series((oof_proba_s >= t).astype(int), index=oof_proba_s.index).shift(1).dropna()
            sig_d = to_daily_filled(sig_w, ret_d.index)
            r     = (sig_d * ret_d).loc[ret_d.index < pd.to_datetime(train_end)].dropna()
            # Sharpe on log returns (OK for small returns)
            v = r.std(ddof=1)
            sharpe_vals.append((r.mean()/v)*np.sqrt(252) if v>0 else np.nan)
        thr_sharpe = float(thr_grid[int(np.nanargmax(sharpe_vals))])
    best = thr_sharpe if thr_sharpe is not None else thr_f1
    return {"thr_f1":thr_f1, "thr_sharpe":thr_sharpe, "best_threshold":float(best)}

def rs(name, est, space, X, y, cv, scoring, refit,
       n_iter=40, random_state=42, n_jobs=1, error_score=np.nan):
    set_seeds(random_state)
    search = RandomizedSearchCV(est, space, n_iter=n_iter, scoring=scoring, refit=refit,
                                cv=cv, n_jobs=n_jobs, random_state=random_state,
                                return_train_score=False, verbose=0, error_score=error_score)
    search.fit(X, y)
    print(f"{name}: best {refit} = {search.best_score_:.4f}")
    return search

# Gating Regime & diagnetics

def regime_from_close(close_d: pd.Series, window: int = 200) -> pd.Series:
    c = to_1d_series(close_d)
    return (c > c.rolling(window).mean()).astype(float)

# Technical indicators

import numpy as np
import pandas as pd

__all__ = [
    "ema", "sma", "rsi", "true_range", "atr", "macd", "adx"
]

def ema(s: pd.Series, span: int) -> pd.Series:

    s = pd.Series(s).astype(float)
    return s.ewm(span=span, adjust=False).mean()

def sma(s: pd.Series, window: int) -> pd.Series:
    s = pd.Series(s).astype(float)
    return s.rolling(window).mean()

def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    c = pd.Series(close).astype(float)
    delta = c.diff()
    up = delta.clip(lower=0.0)
    dn = -delta.clip(upper=0.0)
    # Wiler smoothing via EMA with alpha = 1/n
    avg_up = up.ewm(alpha=1/n, adjust=False).mean()
    avg_dn = dn.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_up / (avg_dn + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    rsi.name = "RSI"
    return rsi

def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    h = pd.Series(high).astype(float)
    l = pd.Series(low).astype(float)
    c = pd.Series(close).astype(float)
    prev_close = c.shift(1)
    a = (h - l).abs()
    b = (h - prev_close).abs()
    c_ = (l - prev_close).abs()
    tr = pd.concat([a, b, c_], axis=1).max(axis=1)
    tr.name = "TR"
    return tr

def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    tr = true_range(high, low, close)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()
    atr_.name = f"ATR{n}"
    return atr_

def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):

    c = pd.Series(close).astype(float)
    fast_ema = c.ewm(span=fast, adjust=False).mean()
    slow_ema = c.ewm(span=slow, adjust=False).mean()
    line = fast_ema - slow_ema
    sig  = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    line.name, sig.name, hist.name = "MACD", "MACDsig", "MACDhist"
    return line, sig, hist

def adx(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 14) -> pd.Series:
    h = pd.Series(high).astype(float)
    l = pd.Series(low).astype(float)
    c = pd.Series(close).astype(float)

    up_move   = h.diff()
    down_move = -l.diff()
    plus_dm  = np.where((up_move > down_move) & (up_move > 0),  up_move,  0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    plus_dm  = pd.Series(plus_dm, index=h.index)
    minus_dm = pd.Series(minus_dm, index=h.index)

    tr  = true_range(h, l, c)
    atr_ = tr.ewm(alpha=1/n, adjust=False).mean()

    plus_di  = 100.0 * (plus_dm.ewm(alpha=1/n, adjust=False).mean()  / (atr_ + 1e-12))
    minus_di = 100.0 * (minus_dm.ewm(alpha=1/n, adjust=False).mean() / (atr_ + 1e-12))
    dx = ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-12)) * 100.0
    adx_ = dx.ewm(alpha=1/n, adjust=False).mean()
    adx_.name = f"ADX{n}"
    return adx_

def to_daily_bfill(sig_w, daily_index):
 
    s = pd.Series(sig_w).sort_index()
    # calendar-day resample 
    s_d = s.resample("D").bfill()
    # align to trading days and forward fill across missing mrkt days
    return s_d.reindex(pd.DatetimeIndex(daily_index)).ffill()

def ann_sharpe(daily_ret: pd.Series) -> float:
    s = pd.Series(daily_ret).dropna()
    v = float(s.std(ddof=1))
    if not np.isfinite(v) or v == 0.0:
        return np.nan
    return float((s.mean() * np.sqrt(252)) / v)
# plotting helper used in Step 6 
def plot_and_table(bt_dict, title, th, tx, test_start):
    import matplotlib.pyplot as plt
    rows = {}
    plt.figure(figsize=(11, 6))
    for name, r in bt_dict.items():
        eq, dd, perf = equity_and_perf(r, start_idx=test_start)
        rows[name] = perf
        eq_sub = eq.loc[eq.index >= pd.to_datetime(test_start)]
        if eq_sub.empty:
            print(f"[warn] {name}: no data after test_start")
            continue
        eq_sub = eq_sub / eq_sub.iloc[0]
        zorder = 10 if name.lower().startswith("buy&hold") else 1
        plt.plot(eq_sub, label=name, lw=2, alpha=0.95, zorder=zorder)
    plt.legend()
    plt.title(f"{title}  |  th={th:.3f}, tx={tx}bps (rebased)")
    plt.tight_layout(); plt.show()
    return pd.DataFrame(rows).T.round(4).sort_values("Sharpe", ascending=False)

# WF used in Step 7 
def walkforward_backtest(model,
                         X_all: pd.DataFrame,
                         y_all: pd.Series,
                         ret_daily: pd.Series,
                         weekly_index: pd.DatetimeIndex,
                         regime: pd.Series = None,
                         refit_every_weeks: int = 13,
                         threshold: float = 0.25,
                         tx_bps: float = 5):
    X_all = X_all.sort_index(); y_all = y_all.sort_index()
    ret_d = to_1d_series(ret_daily).astype(float)
    weeks = pd.DatetimeIndex(weekly_index).sort_values()

    p_all = pd.Series(index=weeks, dtype=float)
    for k in range(0, len(weeks), refit_every_weeks):
        train_end = weeks[k]
        test_blk = weeks[k : k + refit_every_weeks]
        X_tr, y_tr = X_all.loc[:train_end], y_all.loc[:train_end]
        X_te = X_all.loc[test_blk.intersection(X_all.index)]
        # guard: need at least two classes in train
        if X_tr.empty or X_te.empty or y_tr.nunique() < 2:
            continue
        m = clone(model).fit(X_tr, y_tr)
        p_all.loc[X_te.index] = to_proba(m, X_te)

    pos_w = (p_all >= float(threshold)).astype(float).dropna()
    pos_d = to_daily_filled(pos_w, ret_d.index)
    ret_net = backtest_from_positions(ret_d, pos_d, tx_bps=tx_bps, regime=regime)
    return pos_w, ret_net

# alias so Step 7 can call walkforward
walkforward = walkforward_backtest

from sklearn.metrics import average_precision_score
def _ap(est, X, y):
    if hasattr(est, "predict_proba"):
        p = est.predict_proba(X)[:, 1]
    else:
        d = est.decision_function(X); p = (d - d.min())/(d.max() - d.min() + 1e-12)
    return average_precision_score(y, p)


# returns helpers for QuantStats 
def _to_simple(logret: pd.Series, name: str = "") -> pd.Series:
    s = pd.Series(logret).dropna().sort_index().astype(float)
    s = np.expm1(s)                 # simple = exp(log) - 1
    s.name = name or getattr(logret, "name", "")
    s.index = pd.DatetimeIndex(s.index)
    return s

def _clip_bench(bench_simple: pd.Series,
               start=None,
               end=None,
               name: str = "Buy&Hold") -> pd.Series:
    b = pd.Series(bench_simple).dropna().sort_index()
    if start is not None: b = b.loc[pd.to_datetime(start):]
    if end   is not None: b = b.loc[:pd.to_datetime(end)]
    b.name = name
    return b
def print_env_and_config(CONFIG: dict):
    import sys, yfinance as yf, numpy as np, pandas as pd
    from sklearn import __version__ as skl_version
    print("Env summary:")
    print(f"  Python       : {sys.version.split()[0]}")
    print(f"  NumPy        : {np.__version__}")
    print(f"  Pandas       : {pd.__version__}")
    print(f"  scikit-learn : {skl_version}")
    print(f"  yfinance     : {yf.__version__}\n")
    cfg = dict(CONFIG)
    cfg["END"] = cfg.get("END") or "today"
    print("CONFIG:")
    for k, v in cfg.items():
        print(f"  {k}: {v}")
# Feature eng helpers
def build_weekly_features(weekly: pd.DataFrame, H: float = 0.0):
   
    w = weekly.sort_index().copy()
    close = (w["Adj Close"] if "Adj Close" in w else w["Close"]).astype(float)
    high  = w["High"].astype(float)
    low   = w["Low"].astype(float)
    vol   = w["Volume"] if "Volume" in w else None

    # raw weekly log ret and forward label
    r1 = np.log(close).diff()
    y  = (r1.shift(-1) > float(H)).astype(int)  # label uses *future* return

    # technicals (ALL shifted by 1 to avoid lookahead in thj data)
    # RSI / MACD / ATR
    rsi14                = rsi(close, 14).shift(1)
    macd_line, macd_sig, macd_hist = macd(close, fast=12, slow=26, signal=9)
    macd_line, macd_sig, macd_hist = macd_line.shift(1), macd_sig.shift(1), macd_hist.shift(1)
    atr14                = atr(high, low, close, n=14).shift(1)
    atr_pct              = (atr14 / (close + 1e-12)).rename("atr_pct")

    # SMAs, spreads and Bolenger zscore
    sma10 = close.rolling(10).mean()
    sma40 = close.rolling(40).mean()
    sma_spread = ((sma10 - sma40) / (sma40 + 1e-12)).shift(1)

    mid20 = close.rolling(20).mean()
    std20 = close.rolling(20).std(ddof=0)
    bb_z  = ((close - mid20) / (std20 + 1e-12)).shift(1)

    # momentum / carry features
    mom4  = np.log(close).diff(4).shift(1)
    mom12 = np.log(close).diff(12).shift(1)
    ret1_lag = r1.shift(1)

    # Volatility (short & long horizon)
    vol_4w  = r1.rolling(4).std().shift(1)
    vol_13w = r1.rolling(13).std().shift(1)

    # Lagged returns 
    lag2 = r1.shift(2)
    lag4 = r1.shift(4)

    # Interaction terms
    mom_vol = (mom12 * vol_13w).shift(1)
    rsi_bb  = (rsi14 * bb_z).shift(1)

    # Drawdown
    cummax   = close.cummax()
    drawdown = ((close - cummax) / (cummax + 1e-12)).shift(1)

    # Skewnes of returns
    skew_13 = r1.rolling(13).skew().shift(1)

    # volume features (sometimes this was not working so i used a if loop to check availabilty first)
    if vol is not None:
        vol = vol.astype(float)
        vol_z26 = ((vol - vol.rolling(26).mean()) / (vol.rolling(26).std(ddof=0) + 1e-12)).shift(1)
        vol_chg4 = (np.log(vol + 1.0).diff(4)).shift(1)
    else:
        vol_z26 = pd.Series(index=w.index, dtype=float)
        vol_chg4 = pd.Series(index=w.index, dtype=float)

    # Collect features 
    X = pd.DataFrame({
        # original
        "rsi14": rsi14,
        "macd_line": macd_line, "macd_sig": macd_sig, "macd_hist": macd_hist,
        "atr_pct": atr_pct,
        "sma10_40": sma_spread,
        "bb_z20": bb_z,
        "mom4": mom4, "mom12": mom12,
        "ret1_lag": ret1_lag,
        "vol_z26": vol_z26, "vol_chg4": vol_chg4,
        # new
        "vol_4w": vol_4w, "vol_13w": vol_13w,
        "lag2": lag2, "lag4": lag4,
        "mom_vol": mom_vol, "rsi_bb": rsi_bb,
        "drawdown": drawdown, "skew_13": skew_13,
    }, index=w.index)

    # final clean alignment
    X = X.replace([np.inf, -np.inf], np.nan).dropna(how="any")
    y = y.reindex(X.index).astype(int)

    return X, y

# Hysteresis
def hysteresis_from_proba(proba_s: pd.Series, enter: float, exit: float) -> pd.Series:
  
    p = pd.Series(proba_s).astype(float).clip(0, 1)
    if not (0 <= exit < enter <= 1):
        raise ValueError("Require 0 <= exit < enter <= 1 for hysteresis.")
    pos = pd.Series(index=p.index, dtype=float)
    state = 0.0
    for t, v in p.items():
        if state == 0.0 and v >= enter:
            state = 1.0
        elif state == 1.0 and v <= exit:
            state = 0.0
        pos.loc[t] = state
    return pos

# Convenience wraper
def prob_to_pos_with_band(proba_s: pd.Series, enter: float, band: float = 0.10) -> pd.Series:
    exit_th = max(0.0, min(enter - float(band), 0.99))
    return hysteresis_from_proba(proba_s, enter=enter, exit=exit_th)

# Volatility targeting
def scale_position_by_vol(pos_s: pd.Series,
                          ret_daily: pd.Series,
                          target_vol: float = 0.10,
                          lookback: int = 60,
                          lev_cap: float = 3.0) -> pd.Series:

    r = pd.Series(ret_daily).dropna()
    pos = pd.Series(pos_s).reindex(r.index).fillna(0.0)
    # realized vol (anualised)
    ann_vol = r.rolling(lookback).std(ddof=1) * np.sqrt(252)
    lev = (target_vol / ann_vol.replace(0, np.nan)).shift(1)  # 1day delay
    lev = lev.clip(lower=0.0, upper=float(lev_cap)).fillna(0.0)
    return (pos * lev).rename(f"{pos.name or 'pos'}_volT")

# Panel plot + table
def plot_panel(bt_dict: dict,
               title: str = "",
               test_start=None,
               figsize=(11, 6)):
 
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

    rows = {}
    plt.figure(figsize=figsize)

    for name, r in bt_dict.items():
       
        eq, dd, perf = equity_and_perf(r, start_idx=test_start)
        rows[name] = perf

        if not eq.empty:
            # normalize so all curves start at 1 
            eq_plot = eq.copy()
            if test_start is not None:
                eq_plot = eq_plot.loc[eq_plot.index >= pd.to_datetime(test_start)]
            if not eq_plot.empty:
                eq_plot = eq_plot / float(eq_plot.iloc[0])
                plt.plot(eq_plot.index, eq_plot.values, label=name)

    if test_start is not None:
        plt.axvline(pd.to_datetime(test_start), color="k", ls="--", lw=0.7, alpha=0.4)

    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()

    perf_table = pd.DataFrame(rows).T.sort_values("Sharpe", ascending=False).round(4)
    return perf_table


# Deep Learning: Sequence Dataset Helper

def make_timeseries_dataset(X, y, lookback=3, batch_size=64):
   
    ds = timeseries_dataset_from_array(
        X,
        y.shift(-lookback + 1),
        sequence_length=lookback,
        batch_size=batch_size,
        shuffle=False
    )
    return ds

# Deep Learning: LSTM Model Builder

def create_lstm_model(n_features, lookback=3, lr=0.001):
    
    model = Sequential([
        LSTM(128, input_shape=(lookback, n_features), activation='relu', return_sequences=True, name="LSTM_1"),
        Dropout(0.4),
        LSTM(64, activation='relu', name="LSTM_2"),
        Dropout(0.3),
        Dense(1, activation='sigmoid', name="Output")
    ])

    opt = tf.keras.optimizers.Adam(learning_rate=lr)
    model.compile(
        optimizer=opt,
        loss='binary_crossentropy',
        metrics=[AUC(name="AUC"), Precision(name="Precision"), Recall(name="Recall"), 'accuracy']
    )
    return model

# Deep Learning: Evaluation Helper

from sklearn.metrics import precision_score, recall_score, roc_auc_score, accuracy_score

def evaluate_classification(y_true, y_pred, y_proba=None):
   
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0)
    }
    if y_proba is not None:
        metrics["AUC"] = roc_auc_score(y_true, y_proba)
    return metrics

# Deep Learning: Comparison Plot Helper
def plot_metric_comparison(ml_scores, lstm_scores):
    
    df = pd.DataFrame({"Traditional_ML": ml_scores, "LSTM": lstm_scores})
    df.plot(kind="bar", figsize=(6, 4))
    plt.title("Performance Comparison — ML vs LSTM")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()
    return df.round(4)
LOOKBACK = 8
def create_sequences(X, y, lookback=LOOKBACK):
    X_seq, y_seq = [], []
    for i in range(lookback, len(X)):
        X_seq.append(X[i-lookback:i])
        y_seq.append(y[i])
    return np.array(X_seq), np.array(y_seq)

def build_lstm_tunable(hp, lookback, n_features):
   
    from tensorflow.keras import layers, models
    import tensorflow as tf

    units = hp.Int('units', min_value=32, max_value=128, step=32)
    dropout = hp.Choice('dropout', [0.2, 0.3, 0.4, 0.5])
    lr = hp.Choice('lr', [1e-3, 5e-4, 1e-4])

    model = models.Sequential([
        layers.Input(shape=(lookback, n_features)),
        layers.LSTM(units, return_sequences=True),
        layers.BatchNormalization(),
        layers.Dropout(dropout),
        layers.LSTM(units // 2),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.AUC(name="auc")]
    )
    return model

# Properly align LSTM predictions (drop first LOOKBACK weeks)
def neg_f1(w, lstm_norm, ml_norm, y_test_seq, best_thr, best_thr_lstm):
   
    from sklearn.metrics import f1_score
    blended = w * lstm_norm + (1 - w) * ml_norm
    preds = (blended >= (0.5 * (best_thr + best_thr_lstm))).astype(int)
    return -f1_score(y_test_seq, preds)

def to_simple_returns(logret_series):
    return np.exp(logret_series) - 1
def extract_series(bt_output, label):
    if isinstance(bt_output, pd.DataFrame):
        # find appropriate column
        for col in ['Strategy', 'Returns', 'PnL', 'Equity']:
            if col in bt_output.columns:
                s = bt_output[col].copy()
                s.name = label
                return s
        s = bt_output.iloc[:, 0].copy()
        s.name = label
        return s
    elif isinstance(bt_output, pd.Series):
        s = bt_output.copy()
        s.name = label
        return s
def label_for_qs(series, label):
    s = series.copy()
    s.name = label
    df = pd.DataFrame({label: s})
    df[label].index = pd.to_datetime(df[label].index)
    return df[label]

def select_thresholds_lstm(proba_s, y_true_s, ret_daily, verbose=True):
  

    import numpy as np
    import pandas as pd
    from sklearn.metrics import f1_score

    # Align and clean
    data = pd.concat([proba_s, y_true_s], axis=1).dropna()
    data.columns = ["proba", "true"]
    if data.empty:
        raise ValueError("No valid overlapping predictions and labels.")  #Enhanced data handling with help of ChatGPT

    # Adaptive grid based on LSTM probability range
    pmin, pmax = data["proba"].min(), data["proba"].max()
    grid = np.linspace(max(0, pmin - 0.02), min(1, pmax + 0.02), 101)

    f1_list, sharpe_list = [], []

    for thr in grid:
        sig_w = (data["proba"] >= thr).astype(float).shift(1).dropna()

        # F1 score (weekly)
        y_pred = sig_w.reindex(data.index).fillna(method="ffill").fillna(0)
        f1_val = f1_score(data["true"], y_pred)
        f1_list.append(f1_val)

        # Sharpe ratio (daily)
        try:
            sig_d = F.to_daily_filled(sig_w, ret_daily.index)
            strat_ret = (sig_d * ret_daily).dropna()
            if strat_ret.std(ddof=1) > 0:
                sharpe = (strat_ret.mean() / strat_ret.std(ddof=1)) * np.sqrt(252)
            else:
                sharpe = np.nan
        except Exception:
            sharpe = np.nan
        sharpe_list.append(sharpe)

    # Pick best thresholds
    f1_thr = grid[np.nanargmax(f1_list)] if np.any(np.isfinite(f1_list)) else np.nan
    sharpe_thr = grid[np.nanargmax(sharpe_list)] if np.any(np.isfinite(sharpe_list)) else np.nan
    best_thr = sharpe_thr if np.isfinite(sharpe_thr) else f1_thr

    if verbose:
        print(f"[select_thresholds_lstm] F1-opt={f1_thr:.3f}, Sharpe-opt={sharpe_thr:.3f}, BEST={best_thr:.3f}")

    return {
        "best_threshold": float(best_thr),
        "thr_f1": float(f1_thr),
        "thr_sharpe": float(sharpe_thr)
    }
