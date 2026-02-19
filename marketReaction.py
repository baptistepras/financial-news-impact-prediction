#!/usr/bin/env python3
"""
marketReaction.py

Baseline experiment: Impact text -> next-day abnormal return direction (Saudi market).

Final (kept) setup for the write-up:
- No near-zero AR filtering (ar_epsilon = 0). We keep as many labeled events as possible.
- Benchmark index: ^TASI.SR (downloaded separately to avoid Yahoo batch quirks).
- Event label: AR_1d = r_stock(t0->t1) - r_index(t0->t1), where:
    * t0 is the first trading day on/after the announcement date where BOTH index and the stock have a close.
    * t1 is the next trading day after t0 where BOTH have a close.
  y = 1 if AR_1d > 0 else 0.
- Text features: sentence-transformers/all-MiniLM-L6-v2 embeddings (mean pooling).
- Classifier: a single linear layer trained with BCEWithLogitsLoss(pos_weight=neg/pos).
- Split: user-provided train/test CSVs (time split already done upstream), plus an internal
  time-based train/validation split to tune the probability threshold on validation (maximize F1).

Outputs (in --output_dir):
- train_labeled.csv, test_labeled.csv
- training_loss.png
- test_confusion_matrix.png
- test_roc.png
- val_f1_vs_threshold.png
- metrics.json

Run:
  python marketReaction.py --train_csv train_triplets.csv --test_csv test_triplets.csv \
      --output_dir outputs --index_ticker "^TASI.SR"
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
    confusion_matrix,
)

from transformers import AutoModel, AutoTokenizer


# -----------------------------
# Utilities
# -----------------------------

def log(msg: str) -> None:
    print(msg, flush=True)


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_naive_datetime_index(idx: pd.Index) -> pd.DatetimeIndex:
    dt = pd.to_datetime(idx)
    try:
        dt = dt.tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(dt)


# -----------------------------
# Download prices
# -----------------------------

def download_close_series(
    ticker: str,
    start: str,
    end: str,
    max_retries: int,
    sleep_seconds: float,
) -> pd.Series:
    """Download a single ticker close series. Robust for indices."""
    last_err = None
    for k in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
                auto_adjust=False,
                threads=False,
                group_by="column",
            )
            if df is not None and len(df) > 0 and "Close" in df.columns:
                s = df["Close"].dropna()
                if len(s) > 0:
                    s.index = to_naive_datetime_index(s.index)
                    s.name = ticker
                    return s
        except Exception as e:
            last_err = e
        time.sleep(sleep_seconds * (k + 1))
    raise RuntimeError(f"Failed to download index close series for {ticker}. Last error: {last_err}")


def _extract_series_from_download(df: pd.DataFrame, ticker: str, field: str) -> Optional[pd.Series]:
    """Handle different yfinance output layouts."""
    if df is None or len(df) == 0:
        return None

    if isinstance(df.columns, pd.MultiIndex):
        # (Field, Ticker)
        if (field, ticker) in df.columns:
            return df[(field, ticker)]
        # df[field][ticker]
        try:
            if field in df.columns.get_level_values(0):
                sub = df[field]
                if isinstance(sub, pd.DataFrame) and ticker in sub.columns:
                    return sub[ticker]
        except Exception:
            pass
        # df[ticker][field]
        try:
            if ticker in df.columns.get_level_values(0):
                sub = df[ticker]
                if isinstance(sub, pd.DataFrame) and field in sub.columns:
                    return sub[field]
        except Exception:
            pass

    # Single-ticker case
    if field in df.columns:
        return df[field]

    return None


def download_equity_prices(
    tickers: List[str],
    start: str,
    end: str,
    price_field: str,
    chunk_size: int,
    max_retries: int,
    sleep_seconds: float,
) -> Dict[str, pd.Series]:
    """Download multiple equity series in chunks; returns dict ticker->Series."""
    tickers = [t for t in tickers if isinstance(t, str) and t.strip()]
    tickers = list(dict.fromkeys(tickers))

    price_by_ticker: Dict[str, pd.Series] = {}
    failed: List[str] = []

    for i in range(0, len(tickers), chunk_size):
        chunk = tickers[i : i + chunk_size]

        df = None
        last_err = None
        for k in range(max_retries):
            try:
                df = yf.download(
                    chunk,
                    start=start,
                    end=end,
                    progress=False,
                    auto_adjust=False,
                    threads=False,
                    group_by="column",
                )
                break
            except Exception as e:
                last_err = e
                time.sleep(sleep_seconds * (k + 1))

        if df is None:
            log(f"[WARN] Failed to download chunk {i//chunk_size+1} (size={len(chunk)}). Last error: {last_err}")
            failed.extend(chunk)
            continue

        for t in chunk:
            s = _extract_series_from_download(df, t, price_field)
            if s is None:
                failed.append(t)
                continue
            s = s.dropna()
            if len(s) == 0:
                failed.append(t)
                continue
            s.index = to_naive_datetime_index(s.index)
            s.name = t
            price_by_ticker[t] = s

        time.sleep(sleep_seconds)

    if failed:
        uniq_failed = sorted(set(failed))
        log(f"[INFO] {len(uniq_failed)} tickers had no usable '{price_field}' series (missing/delisted/out of range).")

    return price_by_ticker


# -----------------------------
# Labeling
# -----------------------------

def _find_t0_t1(
    event_date: pd.Timestamp,
    stock_dates: pd.DatetimeIndex,
    index_dates: pd.DatetimeIndex,
) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Find t0 (first common trading day >= event_date) and t1 (next common trading day)."""
    if len(stock_dates) == 0 or len(index_dates) == 0:
        return None

    idx_dates = index_dates[index_dates >= event_date]
    if len(idx_dates) == 0:
        return None

    stock_set = set(stock_dates)

    t0 = None
    for d in idx_dates:
        if d in stock_set:
            t0 = pd.Timestamp(d)
            break
    if t0 is None:
        return None

    idx_after = index_dates[index_dates > t0]
    if len(idx_after) == 0:
        return None

    t1 = None
    for d in idx_after:
        if d in stock_set:
            t1 = pd.Timestamp(d)
            break
    if t1 is None:
        return None

    return t0, t1


def label_events(
    triplets: pd.DataFrame,
    price_by_ticker: Dict[str, pd.Series],
    index_close: pd.Series,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Attach (t0,t1) returns, abnormal returns, and binary label."""
    df = triplets.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date", "ticker", "Impact"])

    idx_s = index_close.dropna().copy()
    idx_s.index = to_naive_datetime_index(idx_s.index)
    idx_dates = idx_s.index

    rows = []
    dropped_missing_price = 0

    for r in df.itertuples(index=False):
        date = pd.Timestamp(r.Date).normalize()
        ticker = str(r.ticker)
        impact = str(r.Impact)

        s = price_by_ticker.get(ticker)
        if s is None or len(s) < 3:
            dropped_missing_price += 1
            continue

        stock_s = s.dropna().copy()
        stock_s.index = to_naive_datetime_index(stock_s.index)
        stock_dates = stock_s.index

        tt = _find_t0_t1(date, stock_dates, idx_dates)
        if tt is None:
            dropped_missing_price += 1
            continue
        t0, t1 = tt

        p0 = float(stock_s.loc[t0])
        p1 = float(stock_s.loc[t1])
        p0i = float(idx_s.loc[t0])
        p1i = float(idx_s.loc[t1])

        if p0 <= 0 or p0i <= 0:
            dropped_missing_price += 1
            continue

        r_stock = (p1 - p0) / p0
        r_index = (p1i - p0i) / p0i
        ar_1d = r_stock - r_index
        label = 1 if ar_1d > 0 else 0

        rows.append(
            {
                "Date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "Impact": impact,
                "t0": t0.strftime("%Y-%m-%d"),
                "t1": t1.strftime("%Y-%m-%d"),
                "r_stock_1d": r_stock,
                "r_index_1d": r_index,
                "ar_1d": ar_1d,
                "label": label,
            }
        )

    out = pd.DataFrame(rows)
    meta = {
        "dropped_missing_price": dropped_missing_price,
        "kept": len(out),
    }
    return out, meta


# -----------------------------
# Embeddings
# -----------------------------

@torch.no_grad()
def embed_texts(
    texts: List[str],
    model_name: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    mdl.eval()

    all_vecs = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tok(
            batch,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        out = mdl(**enc).last_hidden_state  # [B, T, H]
        attn = enc["attention_mask"].unsqueeze(-1)  # [B, T, 1]

        masked = out * attn
        summed = masked.sum(dim=1)
        counts = attn.sum(dim=1).clamp(min=1)
        mean = summed / counts
        all_vecs.append(mean.detach().cpu().numpy())

    return np.vstack(all_vecs).astype(np.float32)


# -----------------------------
# Model
# -----------------------------

class LogisticHead(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.linear = nn.Linear(dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x).squeeze(-1)


def train_model(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    X_va: np.ndarray,
    y_va: np.ndarray,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
) -> Tuple[LogisticHead, List[float], List[float], float]:
    set_seed(seed)

    dim = X_tr.shape[1]
    model = LogisticHead(dim).to(device)

    xt = torch.from_numpy(X_tr).to(device)
    yt = torch.from_numpy(y_tr.astype(np.float32)).to(device)
    xv = torch.from_numpy(X_va).to(device)
    yv = torch.from_numpy(y_va.astype(np.float32)).to(device)

    ds = TensorDataset(xt, yt)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    n_pos = float((y_tr == 1).sum())
    n_neg = float((y_tr == 0).sum())
    pos_weight = (n_neg / max(n_pos, 1.0)) if n_pos > 0 else 1.0
    pos_w = torch.tensor([pos_weight], device=device)

    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_state = None
    best_val = float("inf")
    patience = 3
    bad = 0

    for ep in range(1, epochs + 1):
        model.train()
        ep_loss = 0.0
        for xb, yb in dl:
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()
            ep_loss += float(loss.detach().cpu()) * xb.size(0)

        ep_loss /= max(len(ds), 1)
        train_losses.append(ep_loss)

        model.eval()
        with torch.no_grad():
            v_logits = model(xv)
            v_loss = float(crit(v_logits, yv).detach().cpu())
        val_losses.append(v_loss)

        log(f"[INFO] Epoch {ep}/{epochs} - train loss: {ep_loss:.4f} | val loss: {v_loss:.4f}")

        if v_loss < best_val - 1e-4:
            best_val = v_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log(f"[INFO] Early stopping at epoch {ep} (best val loss {best_val:.4f})")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses, float(pos_weight)


def tune_threshold_f1(y_true: np.ndarray, probs: np.ndarray) -> Tuple[float, float, np.ndarray, np.ndarray]:
    thresholds = np.linspace(0.0, 1.0, 101)
    best_thr = 0.5
    best_f1 = -1.0
    f1s = []

    for t in thresholds:
        pred = (probs >= t).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(y_true, pred, average="binary", zero_division=0)
        f1s.append(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(t)

    return best_thr, float(best_f1), thresholds, np.array(f1s, dtype=float)


# -----------------------------
# Plots
# -----------------------------

def plot_training(train_losses: List[float], val_losses: List[float], outpath: Path) -> None:
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="train")
    plt.plot(val_losses, label="val")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray, thr: float, outpath: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["0", "1"])
    plt.yticks([0, 1], ["0", "1"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion matrix (test) | thr={thr:.2f}")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_roc(y_true: np.ndarray, probs: np.ndarray, outpath: Path) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], "--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC curve (test)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_f1_vs_threshold(thresholds: np.ndarray, f1s: np.ndarray, outpath: Path) -> None:
    plt.figure(figsize=(7, 4.5))
    plt.plot(thresholds, f1s)
    plt.xlabel("Threshold")
    plt.ylabel("F1")
    plt.title("F1 vs threshold (validation)")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


# -----------------------------
# CLI / Main
# -----------------------------

@dataclass
class Config:
    train_csv: str
    test_csv: str
    output_dir: str = "outputs"
    index_ticker: str = "^TASI.SR"
    ticker_suffix: Optional[str] = None

    price_field: str = "Close"
    chunk_size: int = 40
    max_retries: int = 3
    sleep_seconds: float = 0.8
    pad_days: int = 3

    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    max_length: int = 256
    embed_batch_size: int = 32

    epochs: int = 15
    lr: float = 2e-3
    weight_decay: float = 1e-2
    batch_size: int = 64
    seed: int = 42
    val_frac: float = 0.15


def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument("--train_csv", required=True)
    p.add_argument("--test_csv", required=True)
    p.add_argument("--output_dir", default="outputs")
    p.add_argument("--index_ticker", default="^TASI.SR")
    p.add_argument("--ticker_suffix", default=None, help="e.g. .SR to enforce benchmark-consistent tickers")
    p.add_argument("--price_field", default="Close")
    p.add_argument("--chunk_size", type=int, default=40)
    p.add_argument("--max_retries", type=int, default=3)
    p.add_argument("--sleep_seconds", type=float, default=0.8)
    p.add_argument("--pad_days", type=int, default=3)
    p.add_argument("--embed_model", default="sentence-transformers/all-MiniLM-L6-v2")
    p.add_argument("--max_length", type=int, default=256)
    p.add_argument("--embed_batch_size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val_frac", type=float, default=0.15)
    a = p.parse_args()
    return Config(**vars(a))


def enforce_suffix(df: pd.DataFrame, suffix: Optional[str]) -> pd.DataFrame:
    if suffix is None:
        return df
    before = len(df)
    out = df[df["ticker"].astype(str).str.endswith(suffix)].copy()
    log(f"[INFO] Enforced ticker suffix '{suffix}': {before}->{len(out)}")
    return out


def time_based_train_val_split(df: pd.DataFrame, val_frac: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = df.copy()
    df["t0"] = pd.to_datetime(df["t0"])
    uniq = np.array(sorted(df["t0"].unique()))
    if len(uniq) < 5:
        m = int((1 - val_frac) * len(df))
        return df.iloc[:m].copy(), df.iloc[m:].copy()

    cut = int((1 - val_frac) * len(uniq))
    cut = max(1, min(cut, len(uniq) - 1))
    cutoff = uniq[cut]
    tr = df[df["t0"] < cutoff].copy()
    va = df[df["t0"] >= cutoff].copy()
    return tr, va


def main() -> None:
    cfg = parse_args()
    outdir = ensure_dir(cfg.output_dir)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[INFO] Device: {device}")

    train = pd.read_csv(cfg.train_csv)
    test = pd.read_csv(cfg.test_csv)

    suffix = cfg.ticker_suffix
    if suffix is None and isinstance(cfg.index_ticker, str) and cfg.index_ticker.endswith(".SR"):
        suffix = ".SR"

    train = enforce_suffix(train, suffix)
    test = enforce_suffix(test, suffix)

    log(f"[INFO] Loaded train triplets: {train.shape}")
    log(f"[INFO] Loaded test triplets:  {test.shape}")

    all_dates = pd.to_datetime(pd.concat([train["Date"], test["Date"]], ignore_index=True), errors="coerce").dropna()
    if len(all_dates) == 0:
        raise RuntimeError("No valid dates found in input CSVs.")

    start = (all_dates.min() - pd.Timedelta(days=cfg.pad_days)).strftime("%Y-%m-%d")
    end = (all_dates.max() + pd.Timedelta(days=cfg.pad_days)).strftime("%Y-%m-%d")

    tickers_all = sorted(set(train["ticker"].astype(str)).union(set(test["ticker"].astype(str))))
    log(f"[INFO] Downloading equity prices ({len(tickers_all)} tickers) from {start} to {end} ...")

    price_by_ticker = download_equity_prices(
        tickers=tickers_all,
        start=start,
        end=end,
        price_field=cfg.price_field,
        chunk_size=cfg.chunk_size,
        max_retries=cfg.max_retries,
        sleep_seconds=cfg.sleep_seconds,
    )
    log(f"[INFO] Downloaded price series for {len(price_by_ticker)}/{len(tickers_all)} tickers")

    log(f"[INFO] Downloading index ({cfg.index_ticker}) separately from {start} to {end} ...")
    index_close = download_close_series(
        ticker=cfg.index_ticker,
        start=start,
        end=end,
        max_retries=cfg.max_retries,
        sleep_seconds=cfg.sleep_seconds,
    )
    log(f"[INFO] Index rows: {len(index_close)} | range: {index_close.index.min().date()} -> {index_close.index.max().date()}")

    log("[INFO] Labelling train triplets (returns + abnormal returns + label) ...")
    train_lab, meta_tr = label_events(train, price_by_ticker, index_close)
    log(f"[INFO] Labeled train rows: {len(train_lab)} (dropped missing price: {meta_tr['dropped_missing_price']})")

    log("[INFO] Labelling test triplets (returns + abnormal returns + label) ...")
    test_lab, meta_te = label_events(test, price_by_ticker, index_close)
    log(f"[INFO] Labeled test rows: {len(test_lab)} (dropped missing price: {meta_te['dropped_missing_price']})")

    if len(train_lab) < 30 or len(test_lab) < 10:
        log("[WARN] Very small labeled dataset. Metrics will be unstable.")

    train_lab.to_csv(outdir / "train_labeled.csv", index=False)
    test_lab.to_csv(outdir / "test_labeled.csv", index=False)

    tr_df, va_df = time_based_train_val_split(train_lab, cfg.val_frac)
    log(f"[INFO] Train/Val split: {len(tr_df)} / {len(va_df)}")

    log("[INFO] Computing embeddings for Impact text ...")
    X_tr = embed_texts(tr_df["Impact"].astype(str).tolist(), cfg.embed_model, device, cfg.max_length, cfg.embed_batch_size)
    X_va = embed_texts(va_df["Impact"].astype(str).tolist(), cfg.embed_model, device, cfg.max_length, cfg.embed_batch_size)
    X_te = embed_texts(test_lab["Impact"].astype(str).tolist(), cfg.embed_model, device, cfg.max_length, cfg.embed_batch_size)

    y_tr = tr_df["label"].to_numpy(dtype=int)
    y_va = va_df["label"].to_numpy(dtype=int)
    y_te = test_lab["label"].to_numpy(dtype=int)

    log(f"[INFO] Embedding dim: {X_tr.shape[1]}")

    log("[INFO] Training classifier ...")
    model, tr_losses, va_losses, pos_weight = train_model(
        X_tr=X_tr,
        y_tr=y_tr,
        X_va=X_va,
        y_va=y_va,
        device=device,
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        seed=cfg.seed,
    )

    plot_training(tr_losses, va_losses, outdir / "training_loss.png")

    model.eval()
    with torch.no_grad():
        va_logits = model(torch.from_numpy(X_va).to(device)).detach().cpu().numpy()
        va_probs = 1 / (1 + np.exp(-va_logits))

    best_thr, best_f1, thr_grid, f1s = tune_threshold_f1(y_va, va_probs)
    plot_f1_vs_threshold(thr_grid, f1s, outdir / "val_f1_vs_threshold.png")
    log(f"[INFO] Best threshold on VAL: {best_thr:.2f} (F1={best_f1:.4f})")

    log("[INFO] Evaluating on test ...")
    with torch.no_grad():
        te_logits = model(torch.from_numpy(X_te).to(device)).detach().cpu().numpy()
        te_probs = 1 / (1 + np.exp(-te_logits))

    te_pred = (te_probs >= best_thr).astype(int)

    acc = accuracy_score(y_te, te_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_te, te_pred, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(y_te, te_probs)
    except Exception:
        auc = float("nan")

    log("[INFO] Test metrics:")
    log(f"[INFO]   accuracy: {acc:.4f}")
    log(f"[INFO]   precision: {prec:.4f}")
    log(f"[INFO]   recall: {rec:.4f}")
    log(f"[INFO]   f1: {f1:.4f}")
    log(f"[INFO]   roc_auc: {auc:.4f}")

    pred_all1 = np.ones_like(y_te)
    pred_all0 = np.zeros_like(y_te)
    acc1 = accuracy_score(y_te, pred_all1)
    f11 = precision_recall_fscore_support(y_te, pred_all1, average="binary", zero_division=0)[2]
    acc0 = accuracy_score(y_te, pred_all0)
    f10 = precision_recall_fscore_support(y_te, pred_all0, average="binary", zero_division=0)[2]
    log("[INFO] Baselines (test):")
    log(f"[INFO]   always-1 accuracy={acc1:.4f} f1={f11:.4f}")
    log(f"[INFO]   always-0 accuracy={acc0:.4f} f1={f10:.4f}")

    cm = confusion_matrix(y_te, te_pred, labels=[0, 1])
    plot_confusion(cm, best_thr, outdir / "test_confusion_matrix.png")
    if not np.isnan(auc):
        plot_roc(y_te, te_probs, outdir / "test_roc.png")

    metrics = {
        "cfg": asdict(cfg),
        "suffix_enforced": suffix,
        "download_start": start,
        "download_end": end,
        "n_tickers_requested": len(tickers_all),
        "n_tickers_downloaded": len(price_by_ticker),
        "index_rows": int(len(index_close)),
        "train_labeled_rows": int(len(train_lab)),
        "test_labeled_rows": int(len(test_lab)),
        "train_rows": int(len(tr_df)),
        "val_rows": int(len(va_df)),
        "pos_weight": float(pos_weight),
        "val_best_threshold": float(best_thr),
        "val_best_f1": float(best_f1),
        "test_accuracy": float(acc),
        "test_precision": float(prec),
        "test_recall": float(rec),
        "test_f1": float(f1),
        "test_roc_auc": float(auc) if not np.isnan(auc) else None,
        "baseline_always1_accuracy": float(acc1),
        "baseline_always1_f1": float(f11),
        "baseline_always0_accuracy": float(acc0),
        "baseline_always0_f1": float(f10),
        "test_label_rate": float(np.mean(y_te)) if len(y_te) else None,
    }

    with open(outdir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    log(f"[INFO] Saved outputs to: {outdir}")


if __name__ == "__main__":
    main()
