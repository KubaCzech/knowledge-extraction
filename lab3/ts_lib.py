"""Helpers for Lab 3: energy-consumption forecasting with LSTMs and static features.

Conventions
-----------
* Raw data: 15-min readings in local time. Duplicated stamps (DST fall-back) are
  averaged, missing stamps (DST spring-forward) are linearly interpolated, then the
  series is resampled to hourly means.
* Test period = the most recent 365 days of each series. The last VAL_DAYS of the
  training period are held out for early stopping.
* A sample is identified by its forecast origin t0 (index of the first forecast hour).
  The model sees the dynamic input y[t0-L_IN : t0] and predicts y[t0 : t0+H].
  Static features are computed from the LOOKBACK hours before t0, so only
  information available at forecast time is used.
* Every window (lookback + horizon) lies entirely inside one period: training
  windows end before the validation start, validation windows end before the
  test start, and test windows start at least LOOKBACK hours after the test start.
  No input or target window crosses the train/test boundary.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from numpy.lib.stride_tricks import sliding_window_view

SERIES = ["TS0277", "TS3423", "TS3449", "TS4011"]
L_IN = 168        # dynamic input length: one week of hourly values
H = 24            # forecast horizon: 1..24 hours ahead
LOOKBACK = 672    # history used for static features: 28 days
VAL_DAYS = 60


# ----------------------------------------------------------------------------- data
def load_raw(path: str) -> pd.Series:
    df = pd.read_csv(path, header=None, names=["t", "y"], parse_dates=["t"])
    s = df.groupby("t")["y"].mean()                               # DST duplicates
    grid = pd.date_range(s.index.min(), s.index.max(), freq="15min")
    return s.reindex(grid).interpolate(limit_direction="both")    # DST gaps


def load_hourly(data_dir: str, name: str) -> pd.Series:
    s = load_raw(os.path.join(data_dir, f"{name}.csv.gz")).resample("h").mean()
    s.name = name
    return s


def calendar_dynamic(idx: pd.DatetimeIndex) -> np.ndarray:
    """Per-hour calendar encoding (sin/cos of hour, weekday, day of year + weekend flag)."""
    hour, dow, doy = idx.hour.values, idx.dayofweek.values, idx.dayofyear.values
    f = [np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
         np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
         np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
         (dow >= 5).astype(float)]
    return np.stack(f, 1).astype(np.float32)


CAL_NAMES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "doy_sin", "doy_cos", "weekend"]


def static_features(y: np.ndarray, idx: pd.DatetimeIndex, t0s: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Static features for forecast origins t0s, using only y[t0-LOOKBACK:t0] and the calendar of t0.

    y must already be normalized by the training mean.
    """
    win = sliding_window_view(y, LOOKBACK)[t0s - LOOKBACK]        # (n, 672), window ends at t0-1
    n = len(t0s)
    feats, names = [], []

    # 1) calendar of the first forecast step
    cal = calendar_dynamic(idx[t0s])
    feats.append(cal); names += [f"t0_{c}" for c in CAL_NAMES]

    # 2) aggregates over the last 28 days / 7 days / 1 day
    days = win.reshape(n, 28, 24)
    daily_mean = days.mean(2)                                      # (n, 28)
    aggr = np.stack([win.mean(1), win.std(1), win[:, -168:].mean(1), win[:, -24:].mean(1),
                     win[:, -24:].std(1), daily_mean.std(1)], 1)
    feats.append(aggr); names += ["mean_28d", "std_28d", "mean_7d", "mean_1d", "std_1d", "std_daily_means_28d"]

    # 3) percentiles of the last 28 days
    q = [0, 10, 20, 40, 60, 80, 90, 100]
    feats.append(np.percentile(win, q, axis=1).T); names += [f"p{p}_28d" for p in q]

    # 4) trend: slope of the daily means (per day) and ratio of last week vs previous 3 weeks
    x = np.arange(28) - 13.5
    slope = (daily_mean * x).sum(1) / (x ** 2).sum()
    ratio = daily_mean[:, -7:].mean(1) / (daily_mean[:, :-7].mean(1) + 1e-3)
    feats.append(np.stack([slope, ratio], 1)); names += ["trend_slope_28d", "ratio_7d_vs_prev21d"]

    # 5) average daily profile over 28 days, aligned so that profile[k] is the
    #    typical value k hours after t0's hour-of-day (LOOKBACK is a multiple of 24)
    prof = days.mean(1)
    feats.append(prof); names += [f"profile_h{k}" for k in range(24)]

    # 6) weekday / weekend contrast over the lookback window
    wd = idx[t0s - LOOKBACK].dayofweek.values                     # weekday of first lookback day
    dow_days = (wd[:, None] + np.arange(28)[None]) % 7
    is_we = dow_days >= 5
    we = (daily_mean * is_we).sum(1) / is_we.sum(1)
    wk = (daily_mean * ~is_we).sum(1) / (~is_we).sum(1)
    feats.append(np.stack([we / (wk + 1e-3)], 1)); names += ["weekend_weekday_ratio"]

    return np.concatenate(feats, 1).astype(np.float32), names


@dataclass
class SeriesData:
    name: str
    hourly: pd.Series
    mean: float
    y: np.ndarray            # normalized consumption
    cal: np.ndarray          # (N, 7) calendar dynamic features
    test_start: int
    val_start: int
    t0: dict = field(default_factory=dict)   # split -> forecast origins
    static: dict = field(default_factory=dict)


def prepare(data_dir: str, name: str) -> SeriesData:
    h = load_hourly(data_dir, name)
    idx = h.index
    test_start_ts = idx[-1] + pd.Timedelta(hours=1) - pd.Timedelta(days=365)
    test_start = int(np.searchsorted(idx, test_start_ts))
    val_start = test_start - VAL_DAYS * 24
    mean = float(h.values[:test_start].mean())                    # training data only
    y = (h.values / mean).astype(np.float32)
    d = SeriesData(name, h, mean, y, calendar_dynamic(idx), test_start, val_start)
    N = len(y)
    d.t0["train"] = np.arange(LOOKBACK, val_start - H + 1)
    d.t0["val"] = np.arange(max(val_start, LOOKBACK), test_start - H + 1)   # lookback may use train history
    d.t0["test"] = np.arange(test_start + LOOKBACK, N - H + 1)              # strictly inside the test year
    for k, t0s in d.t0.items():
        d.static[k], _ = static_features(y, idx, t0s)
    return d


def static_names(d: SeriesData) -> list[str]:
    return static_features(d.y, d.hourly.index, d.t0["test"][:1])[1]


# ---------------------------------------------------------------------------- models
class Forecaster(nn.Module):
    """LSTM encoder -> MLP head predicting H future values at once.

    static_mode:
      none   - dynamic inputs only
      seq    - static vector repeated at every time step as extra input (naive baseline)
      concat - latent(static) concatenated with the last hidden state      (Miebs et al. #1)
      init   - latent(static) initializes h0 and c0 of all LSTM layers     (Miebs et al. #2)
      hybrid - init + concat                                                (Miebs et al. #3)
    All variants share a linear skip connection initialized to the weekly-naive forecast.

      film   - extension: init + the latent produces a feature-wise affine
               modulation (FiLM: h * (1 + gamma) + beta) of the dynamic input at every
               time step and of the final hidden state, i.e. the static context
               re-weights the dynamic channels instead of only being appended.
    """

    def __init__(self, n_dyn, n_static=0, static_mode="none", hidden=64, layers=1, latent=32,
                 dropout=0.1):
        super().__init__()
        self.mode, self.hidden, self.layers = static_mode, hidden, layers
        rnn_in = n_dyn + (n_static if static_mode == "seq" else 0)
        self.lstm = nn.LSTM(rnn_in, hidden, layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        if static_mode == "film":
            self.film_in = nn.Linear(latent, 2 * n_dyn)
            self.film_out = nn.Linear(latent, 2 * hidden)
            for l in (self.film_in, self.film_out):   # start as identity modulation
                nn.init.zeros_(l.weight); nn.init.zeros_(l.bias)
        if static_mode in ("concat", "init", "hybrid", "film"):
            self.enc = nn.Sequential(nn.Linear(n_static, latent), nn.ReLU(), nn.Linear(latent, latent), nn.Tanh())
        if static_mode in ("init", "hybrid", "film"):
            self.to_h = nn.Linear(latent, layers * hidden)
            self.to_c = nn.Linear(latent, layers * hidden)
        head_in = hidden + (latent if static_mode in ("concat", "hybrid") else 0)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(head_in, 128), nn.ReLU(), nn.Linear(128, H))
        # linear "highway" from the raw consumption input to the outputs (as in LSTNet), initialized
        # to the weekly seasonal-naive forecast y[t0+k-168]; the LSTM branch learns the correction.
        self.skip = nn.Linear(L_IN, H)
        with torch.no_grad():
            self.skip.weight.zero_(); self.skip.bias.zero_()
            self.skip.weight[torch.arange(H), torch.arange(H)] = 1.0

    def _init_state(self, z):
        B = z.shape[0]
        h0 = torch.tanh(self.to_h(z)).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        c0 = self.to_c(z).view(B, self.layers, self.hidden).transpose(0, 1).contiguous()
        return h0, c0

    def forward(self, x, s=None):
        m = self.mode
        base = self.skip(x[:, :, 0])            # channel 0 is always the consumption
        if m == "seq":
            x = torch.cat([x, s[:, None, :].expand(-1, x.shape[1], -1)], 2)
        z = self.enc(s) if hasattr(self, "enc") else None
        if m == "film":
            g, b = self.film_in(z).chunk(2, 1)
            x = x * (1 + g[:, None]) + b[:, None]
        state = self._init_state(z) if m in ("init", "hybrid", "film") else None
        out, _ = self.lstm(x, state)
        last = out[:, -1]
        if m == "film":
            g, b = self.film_out(z).chunk(2, 1)
            last = last * (1 + g) + b
        if m in ("concat", "hybrid"):
            last = torch.cat([last, z], 1)
        return base + self.head(last)


# --------------------------------------------------------------------------- batching
class Pool:
    """Concatenates several series so that one model can be trained on all of them."""

    def __init__(self, series: list[SeriesData], use_cal: bool, static_mu=None, static_sd=None):
        self.series = series
        ys, dyn, self.offset = [], [], []
        off = 0
        for d in series:
            self.offset.append(off)
            f = d.y[:, None] if not use_cal else np.concatenate([d.y[:, None], d.cal], 1)
            dyn.append(f); ys.append(d.y); off += len(d.y)
        self.dyn = torch.from_numpy(np.concatenate(dyn)).float()
        self.y = torch.from_numpy(np.concatenate(ys)).float()
        self.n_dyn = self.dyn.shape[1]
        if static_mu is None:  # standardize static features with training statistics
            tr = np.concatenate([d.static["train"] for d in series])
            static_mu, static_sd = tr.mean(0), tr.std(0) + 1e-6
        self.static_mu, self.static_sd = static_mu, static_sd
        self.n_static = len(static_mu)

    def split(self, name):
        g = np.concatenate([d.t0[name] + o for d, o in zip(self.series, self.offset)])
        s = np.concatenate([(d.static[name] - self.static_mu) / self.static_sd for d in self.series])
        return torch.from_numpy(g).long(), torch.from_numpy(s.astype(np.float32))

    def batch(self, g0, s):
        xi = g0[:, None] + torch.arange(-L_IN, 0)[None]
        yi = g0[:, None] + torch.arange(0, H)[None]
        return self.dyn[xi], s, self.y[yi]


# --------------------------------------------------------------------------- training
def predict(model, pool, g, s, bs=2048):
    model.eval(); out = []
    with torch.no_grad():
        for i in range(0, len(g), bs):
            x, ss, _ = pool.batch(g[i:i + bs], s[i:i + bs])
            out.append(model(x, ss))
    return torch.cat(out).numpy()


def train_model(pool: Pool, static_mode="none", epochs=15, patience=4, bs=256, lr=2e-3,
                seed=0, stride=3, max_train=None, verbose=True, **kw):
    """Adam + MSE, ReduceLROnPlateau, early stopping on the validation period.

    stride: use every `stride`-th forecast origin for training (neighbouring hourly windows
    are almost identical, so this cuts the cost with little loss of information).
    """
    torch.manual_seed(seed); np.random.seed(seed)
    model = Forecaster(pool.n_dyn, pool.n_static, static_mode, **kw)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=0.5, patience=2)
    gtr, str_ = pool.split("train"); gva, sva = pool.split("val")
    gtr, str_ = gtr[::stride], str_[::stride]
    yva = pool.y[gva[:, None] + torch.arange(H)[None]].numpy()
    best, best_state, wait, hist = np.inf, None, 0, []
    for ep in range(epochs):
        t = time.time(); model.train()
        perm = torch.randperm(len(gtr))
        if max_train: perm = perm[:max_train]
        tot = 0.0
        for i in range(0, len(perm), bs):
            b = perm[i:i + bs]
            x, s, y = pool.batch(gtr[b], str_[b])
            loss = nn.functional.mse_loss(model(x, s), y)
            opt.zero_grad(); loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); tot += loss.item() * len(b)
        val = float(((predict(model, pool, gva, sva) - yva) ** 2).mean())
        sched.step(val)
        hist.append((tot / len(perm), val))
        if verbose:
            print(f"  ep {ep:2d} train {tot/len(perm):.4f} val {val:.4f} ({time.time()-t:.0f}s)", flush=True)
        if val < best - 1e-5:
            best, wait = val, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience: break
    model.load_state_dict(best_state)
    return model, hist


def evaluate(model, pool: Pool, split="test") -> dict[str, pd.DataFrame]:
    """Per-series, per-horizon metrics in normalized units (i.e. relative to the training mean)."""
    res = {}
    for d, o in zip(pool.series, pool.offset):
        g = torch.from_numpy(d.t0[split] + o).long()
        s = torch.from_numpy(((d.static[split] - pool.static_mu) / pool.static_sd).astype(np.float32))
        p = predict(model, pool, g, s)
        y = d.y[d.t0[split][:, None] + np.arange(H)[None]]
        res[d.name] = horizon_metrics(p, y)
    return res


def horizon_metrics(p: np.ndarray, y: np.ndarray) -> pd.DataFrame:
    e = p - y
    return pd.DataFrame({"horizon": np.arange(1, H + 1),
                         "RMSE": np.sqrt((e ** 2).mean(0)),
                         "MAE": np.abs(e).mean(0)})


def naive_metrics(d: SeriesData, lag: int, split="test") -> pd.DataFrame:
    """Seasonal-naive forecast y_hat[t0+k] = y[t0+k-lag] (lag >= H so it only uses the input)."""
    t0 = d.t0[split][:, None] + np.arange(H)[None]
    return horizon_metrics(d.y[t0 - lag], d.y[t0])


def save_metrics(path, metrics: dict[str, pd.DataFrame], hist=None):
    out = {k: v.to_dict(orient="list") for k, v in metrics.items()}
    if hist is not None: out["_history"] = hist
    with open(path, "w") as f: json.dump(out, f)


def load_metrics(path) -> dict[str, pd.DataFrame]:
    with open(path) as f: raw = json.load(f)
    return {k: pd.DataFrame(v) for k, v in raw.items() if not k.startswith("_")}


# ----------------------------------------------------------------------- experiments
def midnight_predictions(model, pool: Pool) -> dict[str, np.ndarray]:
    """Day-ahead forecasts issued at 00:00 of every test day (for plots)."""
    out = {}
    for d, o in zip(pool.series, pool.offset):
        sel = d.hourly.index[d.t0["test"]].hour == 0
        t0 = d.t0["test"][sel]
        s = ((d.static["test"][sel] - pool.static_mu) / pool.static_sd).astype(np.float32)
        out[d.name] = np.stack([t0.astype(np.float32),
                                *predict(model, pool, torch.from_numpy(t0 + o).long(), torch.from_numpy(s)).T], 1)
    return out


def run_experiment(tag, series: list[SeriesData], use_cal: bool, static_mode="none",
                   results_dir="results", retrain=False, **kw):
    """Train + evaluate one configuration; cached in results_dir/<tag>.json (+ .npz, .pt)."""
    os.makedirs(results_dir, exist_ok=True)
    base = os.path.join(results_dir, tag)
    if os.path.exists(base + ".json") and not retrain:
        with open(base + ".json") as f: raw = json.load(f)
        metrics = {k: pd.DataFrame(v) for k, v in raw.items() if not k.startswith("_")}
        preds = dict(np.load(base + ".npz"))
        return metrics, raw["_history"], preds
    pool = Pool(series, use_cal)
    t = time.time()
    print(f"[{tag}] training ({static_mode}, calendar={use_cal}, series={[d.name for d in series]})", flush=True)
    model, hist = train_model(pool, static_mode, **kw)
    metrics = evaluate(model, pool)
    preds = midnight_predictions(model, pool)
    out = {k: v.to_dict(orient="list") for k, v in metrics.items()}
    out["_history"] = hist
    out["_meta"] = {"static_mode": static_mode, "use_cal": use_cal, "seconds": time.time() - t,
                    "n_params": sum(p.numel() for p in model.parameters())}
    with open(base + ".json", "w") as f: json.dump(out, f)
    np.savez_compressed(base + ".npz", **preds)
    torch.save(model.state_dict(), base + ".pt")
    print(f"[{tag}] done in {time.time()-t:.0f}s", flush=True)
    return metrics, hist, preds
