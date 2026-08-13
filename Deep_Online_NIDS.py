
from __future__ import annotations

import logging
import os
import pickle
import random
import re
import time
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from river import compose
from river import metrics as river_metrics
from river import preprocessing as river_pp
from river.drift import KSWIN
import deep_river.classification as drc

logger = logging.getLogger("ids_pipeline")


# ──────────────────────────────────────────────────────────────────────────
# 1. Config
# ──────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Config:
    seed: int = 42
    device: torch.device = torch.device("cpu")
    file_path: str = "Edge-IIoTset.csv"

    label_mapping: Dict[str, int] = field(
        default_factory=lambda: {
            "Normal": 0, "DDoS": 1, "Injection": 2,
            "Malware": 3, "Scanning": 4, "MITM": 5,
        }
    )

    stream_delay_s: float = 0.0
    warmup_frac: float = 0.3

    # KSWIN drift detection
    kswin_window: int = 200
    kswin_alpha: float = 0.001

    # Adaptation
    lr_base: float = 1e-3
    lr_spike: float = 5e-3
    adapt_window: int = 500
    reservoir_size: int = 200
    replay_rounds: int = 3
    pseudo_label_threshold: float = 0.85

    # Model architecture
    hidden_dims: Tuple[int, int] = (128, 64)
    dropout: float = 0.10
    grad_clip: float = 5.0

    latency_every: int = 10  

    @property
    def inv_label(self) -> Dict[int, str]:
        return {v: k for k, v in self.label_mapping.items()}

    @property
    def n_classes(self) -> int:
        return len(self.label_mapping)

    @property
    def class_names(self) -> List[str]:
        inv = self.inv_label
        return [inv[i] for i in range(self.n_classes)]

    @property
    def normal_idx(self) -> int:
        return self.label_mapping["Normal"]


def set_global_seed(cfg: Config) -> None:
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)


# ──────────────────────────────────────────────────────────────────────────
# 2. Data loading
# ──────────────────────────────────────────────────────────────────────────
def load_and_split(cfg: Config) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    df = pd.read_csv(cfg.file_path)
    logger.info("Loaded: %s rows x %s cols", f"{df.shape[0]:,}", df.shape[1])

    df.columns = [re.sub(r"[^0-9a-zA-Z_]", "_", c) for c in df.columns]
    df["Attack_type"] = df["Attack_type"].map(cfg.label_mapping)
    if df["Attack_type"].isna().any():
        raise ValueError("Unknown label — check LABEL_MAPPING")
    df["Attack_type"] = df["Attack_type"].astype(np.int64)

    feature_cols = [c for c in df.columns if c != "Attack_type"]

    df = df.sample(frac=1, random_state=cfg.seed).reset_index(drop=True)
    split = int(len(df) * cfg.warmup_frac)
    train_df = df.iloc[:split].reset_index(drop=True)
    test_df = df.iloc[split:].reset_index(drop=True)

    logger.info("Warm-up: %s labeled | Stream: %s (post-hoc only)",
                f"{len(train_df):,}", f"{len(test_df):,}")
    return train_df, test_df, feature_cols


# ──────────────────────────────────────────────────────────────────────────
# 3. Stream simulators
# ──────────────────────────────────────────────────────────────────────────
def labeled_stream(
    dataframe: pd.DataFrame, feat_cols: List[str], delay_s: float = 0.0
) -> Iterator[Tuple[int, Dict, int]]:
    label_pos = dataframe.columns.get_loc("Attack_type")
    cols = feat_cols
    for i, row in enumerate(dataframe[cols + ["Attack_type"]].itertuples(index=False, name=None)):
        x = dict(zip(cols, row[:-1]))
        y = int(row[-1])
        if delay_s > 0:
            time.sleep(delay_s)
        yield i, x, y


def unlabeled_stream(
    dataframe: pd.DataFrame, feat_cols: List[str], delay_s: float = 0.0
) -> Iterator[Tuple[int, Dict]]:
    for i, row in enumerate(dataframe[feat_cols].itertuples(index=False, name=None)):
        x = dict(zip(feat_cols, row))
        if delay_s > 0:
            time.sleep(delay_s)
        yield i, x


# ──────────────────────────────────────────────────────────────────────────
# 4. Model
# ──────────────────────────────────────────────────────────────────────────
class TabularMLP(nn.Module):
    def __init__(self, n_features: int, n_classes: int, hidden_dims: Tuple[int, int], dropout: float):
        super().__init__()
        h1, h2 = hidden_dims
        self.network = nn.Sequential(
            nn.Linear(n_features, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
            nn.Linear(h2, n_classes),
        )

    def forward(self, x):
        return self.network(x)


def build_model(cfg: Config, train_df: pd.DataFrame, n_features: int):
    counts = np.bincount(train_df["Attack_type"].to_numpy(), minlength=cfg.n_classes).astype(float)
    weights = counts.sum() / (cfg.n_classes * np.maximum(counts, 1.0))
    weights = np.clip(weights, 0.25, 8.0).astype("float32")
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=cfg.device))

    common_kwargs = dict(
        loss_fn=loss_fn, optimizer_fn="adam", lr=cfg.lr_base,
        is_class_incremental=False, device=str(cfg.device), seed=cfg.seed,
    )

    torch_module = TabularMLP(n_features, cfg.n_classes, cfg.hidden_dims, cfg.dropout)
    n_params = sum(p.numel() for p in torch_module.parameters())  

    if hasattr(drc.Classifier, "initialize_module"):
        deep_river_api = "legacy-lazy"
        classifier = drc.Classifier(
            module=TabularMLP,
            n_features=n_features,
            n_classes=cfg.n_classes,
            hidden_dims=cfg.hidden_dims,
            dropout=cfg.dropout,
            **common_kwargs,
        )
    else:
        deep_river_api = "modern-eager"
        classifier = drc.Classifier(
            module=torch_module, gradient_clip_value=cfg.grad_clip, **common_kwargs,
        )

    classifier.observed_classes |= set(range(cfg.n_classes))
    pipeline = compose.Pipeline(river_pp.StandardScaler(), classifier)
    pipeline._deep_river_api = deep_river_api
    pipeline._n_params = n_params  
    return pipeline


def set_lr(pipeline, value: float) -> None:
    classifier = pipeline[-1]
    if getattr(classifier, "optimizer", None) is not None:
        for group in classifier.optimizer.param_groups:
            group["lr"] = value


def run_compat_gate(pipeline, train_df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> None:
    
    x0 = {name: float(train_df.iloc[0][name]) for name in feature_cols}
    proba = pipeline.predict_proba_one(x0)
    keys, expected = set(proba.keys()), set(range(cfg.n_classes))
    if keys != expected:
        raise RuntimeError(f"deep-river compatibility gate failed: got {sorted(map(str, keys))}, "
                            f"expected {sorted(expected)}")
    if len(proba) != cfg.n_classes or not np.isclose(sum(proba.values()), 1.0, atol=1e-5):
        raise RuntimeError("deep-river compatibility gate returned invalid probabilities.")
    logger.info("deep-river compatibility gate passed (API=%s).", pipeline._deep_river_api)


# ──────────────────────────────────────────────────────────────────────────
# 5. Drift detection utilities
# ──────────────────────────────────────────────────────────────────────────
def predict_with_uncertainty(pipeline, x_dict: Dict) -> Tuple[Optional[int], Optional[float], Optional[Dict]]:
    
    proba = pipeline.predict_proba_one(x_dict)
    if not proba:
        return None, None, None
    pred = max(proba, key=proba.get)
    uncertainty = 1.0 - proba[pred]
    return pred, uncertainty, proba


class PerClassReservoir:

    def __init__(self, n_classes: int, capacity: int):
        self.capacity = capacity
        self.buffers: List[List[Tuple[Dict, int]]] = [[] for _ in range(n_classes)]
        self.counts = [0] * n_classes

    def add(self, x: Dict, y: int) -> None:
        buf = self.buffers[y]
        self.counts[y] += 1
        if len(buf) < self.capacity:
            buf.append((x, y))
        else:
            j = random.randint(0, self.counts[y] - 1)
            if j < self.capacity:
                buf[j] = (x, y)

    def sample_all(self) -> List[Tuple[Dict, int]]:
        all_s = [s for buf in self.buffers for s in buf]
        random.shuffle(all_s)
        return all_s

    def summary(self, class_names: List[str]) -> Dict[str, int]:
        return {class_names[i]: len(b) for i, b in enumerate(self.buffers)}


# ──────────────────────────────────────────────────────────────────────────
# 6. Phase 1 — Warm-up (labeled, supervised)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class WarmupResult:
    warmup_metric: river_metrics.Accuracy
    warmup_acc_log: List[Tuple[int, float]]
    warmup_lat_ns: List[int]
    warmup_drifts: List[Dict]
    peak_mem_mb: float
    elapsed_s: float


def run_warmup(pipeline, det_kswin: KSWIN, reservoir: PerClassReservoir,
                train_df: pd.DataFrame, feature_cols: List[str], cfg: Config) -> WarmupResult:
    warmup_metric = river_metrics.Accuracy()
    warmup_acc_log: List[Tuple[int, float]] = []
    warmup_lat_ns: List[int] = []
    warmup_drifts: List[Dict] = []

    process = psutil.Process(os.getpid())
    peak_mem = 0.0
    wu_adapt_countdown, wu_is_adapting = 0, False
    log_iv = max(1, len(train_df) // 20)

    logger.info("Phase 1 — Warm-up on %s labeled packets ...", f"{len(train_df):,}")
    t0 = time.perf_counter_ns()

    for i, x, y in labeled_stream(train_df, feature_cols, delay_s=cfg.stream_delay_s):
        reservoir.add(x, y)

        yp, u, _ = predict_with_uncertainty(pipeline, x)
        if yp is not None:
            warmup_metric.update(y, yp)

        if u is not None:
            det_kswin.update(u)
            if det_kswin.drift_detected and not wu_is_adapting:
                logger.info("[WARM-UP DRIFT] @ packet %s acc=%.4f uncertainty=%.4f",
                            f"{i:,}", warmup_metric.get(), u)
                set_lr(pipeline, cfg.lr_spike)
                wu_is_adapting, wu_adapt_countdown = True, cfg.adapt_window
                warmup_drifts.append({"packet_index": i, "phase": "warmup",
                                       "accuracy": warmup_metric.get(), "uncertainty": u})

        t_s = time.perf_counter_ns()
        pipeline.learn_one(x, y)
        t_e = time.perf_counter_ns()

        if wu_is_adapting:
            wu_adapt_countdown -= 1
            if wu_adapt_countdown <= 0:
                set_lr(pipeline, cfg.lr_base)
                wu_is_adapting = False

        if i % cfg.latency_every == 0:
            warmup_lat_ns.append(t_e - t_s)
            peak_mem = max(peak_mem, process.memory_info().rss / 1024 ** 2)

        warmup_acc_log.append((i, warmup_metric.get()))

        if (i + 1) % log_iv == 0:
            logger.info("[%s/%s] acc=%.4f wu_drifts=%d",
                        f"{i+1:>6,}", f"{len(train_df):,}", warmup_metric.get(), len(warmup_drifts))

    elapsed_s = (time.perf_counter_ns() - t0) / 1e9
    logger.info("Warm-up done in %.1fs | acc=%.4f | drifts=%d",
                elapsed_s, warmup_metric.get(), len(warmup_drifts))
    return WarmupResult(warmup_metric, warmup_acc_log, warmup_lat_ns, warmup_drifts, peak_mem, elapsed_s)


# ──────────────────────────────────────────────────────────────────────────
# 7. Phase 2 — Live stream (unlabeled, drift-triggered adaptation)
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class StreamResult:
    y_pred_list: List[int]
    uncertainty_log: List[Tuple[int, float]]
    stream_drifts: List[Dict]
    infer_lat_ns: List[int]
    peak_mem_mb: float
    elapsed_s: float


def run_stream(pipeline, det_kswin: KSWIN, reservoir: PerClassReservoir,
                test_df: pd.DataFrame, feature_cols: List[str], cfg: Config,
                start_peak_mem: float, global_start: int) -> StreamResult:
    y_pred_list: List[int] = []
    uncertainty_log: List[Tuple[int, float]] = []
    stream_drifts: List[Dict] = []
    infer_lat_ns: List[int] = []

    process = psutil.Process(os.getpid())
    peak_mem = start_peak_mem
    adapt_countdown, is_adapting = 0, False
    global_i = global_start
    log_iv = max(1, len(test_df) // 20)

    logger.info("Phase 2 — Live unlabeled stream on %s packets ...", f"{len(test_df):,}")
    t0 = time.perf_counter_ns()

    for i, x in unlabeled_stream(test_df, feature_cols, delay_s=cfg.stream_delay_s):
        t_i0 = time.perf_counter_ns()

        yp, u, proba = predict_with_uncertainty(pipeline, x)
        t_i1 = time.perf_counter_ns()

        if i % cfg.latency_every == 0:
            infer_lat_ns.append(t_i1 - t_i0)
            peak_mem = max(peak_mem, process.memory_info().rss / 1024 ** 2)

        y_pred_list.append(yp if yp is not None else -1)

        if u is not None:
            uncertainty_log.append((global_i, u))
            det_kswin.update(u)

            if det_kswin.drift_detected and not is_adapting:
                logger.info("DRIFT [KSWIN] @ packet %s uncertainty=%.4f", f"{global_i:,}", u)
                set_lr(pipeline, cfg.lr_spike)
                is_adapting, adapt_countdown = True, cfg.adapt_window

                replay = reservoir.sample_all()
                t_r0 = time.perf_counter_ns()
                for _ in range(cfg.replay_rounds):
                    random.shuffle(replay)
                    for rx, ry in replay:
                        pipeline.learn_one(rx, ry)
                t_r1 = time.perf_counter_ns()

                stream_drifts.append({
                    "packet_index": global_i, "phase": "stream", "uncertainty": u,
                    "replay_samples": len(replay), "replay_time_ms": (t_r1 - t_r0) / 1e6,
                })

        if is_adapting and yp is not None and proba and max(proba.values()) >= cfg.pseudo_label_threshold:
            pipeline.learn_one(x, yp)
        if is_adapting:
            adapt_countdown -= 1
            if adapt_countdown <= 0:
                set_lr(pipeline, cfg.lr_base)
                is_adapting = False
                logger.info("Adaptation closed @ packet %s", f"{global_i:,}")

        global_i += 1
        if (i + 1) % log_iv == 0:
            logger.info("[%s/%s] drifts=%d adapting=%s",
                        f"{i+1:>7,}", f"{len(test_df):,}", len(stream_drifts), is_adapting)

    elapsed_s = (time.perf_counter_ns() - t0) / 1e9
    logger.info("Stream done in %.1fs | drifts=%d", elapsed_s, len(stream_drifts))
    return StreamResult(y_pred_list, uncertainty_log, stream_drifts, infer_lat_ns, peak_mem, elapsed_s)


# ──────────────────────────────────────────────────────────────────────────
# 8. Post-hoc evaluation
# ──────────────────────────────────────────────────────────────────────────
@dataclass
class EvalResult:
    acc: float
    macro_prec: float
    macro_rec: float
    macro_f1: float
    weighted_prec: float
    weighted_rec: float
    weighted_f1: float
    cm: np.ndarray
    report: dict
    det_rate: float
    fpr: float
    bin_f1: float
    stream_acc_log: List[Tuple[int, float]]


def evaluate_post_hoc(y_true_test: List[int], y_pred_list: List[int], cfg: Config,
                       train_len: int) -> EvalResult:
    y_true = np.asarray(y_true_test)
    y_pred = np.asarray(y_pred_list)

    acc = accuracy_score(y_true, y_pred)
    macro_prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
    macro_rec = recall_score(y_true, y_pred, average="macro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    weighted_prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_rec = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    weighted_f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(cfg.n_classes)))
    report = classification_report(y_true, y_pred, labels=list(range(cfg.n_classes)),
                                    target_names=cfg.class_names, output_dict=True, zero_division=0)

    y_bin_t = (y_true != cfg.normal_idx).astype(int)
    y_bin_p = (y_pred != cfg.normal_idx).astype(int)
    bin_cm = confusion_matrix(y_bin_t, y_bin_p)
    TN, FP, FN, TP = bin_cm.ravel() if bin_cm.size == 4 else (0, 0, 0, 0)
    det_rate = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    fpr = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    bin_f1 = f1_score(y_bin_t, y_bin_p, zero_division=0)

    correct = np.cumsum(y_true == y_pred)
    running_acc = correct / np.arange(1, len(y_true) + 1)
    stream_acc_log = list(zip(train_len + np.arange(len(y_true)), running_acc))

    return EvalResult(acc, macro_prec, macro_rec, macro_f1, weighted_prec, weighted_rec,
                       weighted_f1, cm, report, det_rate, fpr, bin_f1, stream_acc_log)


# ──────────────────────────────────────────────────────────────────────────
# 9. Edge metrics + persistence
# ──────────────────────────────────────────────────────────────────────────
def compute_edge_metrics(pipeline, warmup: WarmupResult, stream: StreamResult,
                          train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: Config,
                          model_path: str = "online_ids_model.pkl") -> dict:
    with open(model_path, "wb") as f:
        pickle.dump(pipeline, f)
    model_size_kb = os.path.getsize(model_path) / 1024

    infer_lat_ms = np.array(stream.infer_lat_ns) / 1e6
    train_lat_ms = np.array(warmup.warmup_lat_ns) / 1e6

    all_drifts = warmup.warmup_drifts + stream.stream_drifts
    peak_mem = max(warmup.peak_mem_mb, stream.peak_mem_mb)

    return {
        "mean_inference_latency_ms": float(np.mean(infer_lat_ms)),
        "p95_inference_latency_ms": float(np.percentile(infer_lat_ms, 95)),
        "mean_training_latency_ms": float(np.mean(train_lat_ms)),
        "training_throughput_sps": len(train_df) / warmup.elapsed_s,
        "testing_throughput_sps": len(test_df) / stream.elapsed_s,
        "peak_memory_mb": peak_mem,
        "model_size_kb": model_size_kb,
        "trainable_parameters": pipeline._n_params,
        "warmup_drift_events": len(warmup.warmup_drifts),
        "stream_drift_events": len(stream.stream_drifts),
        "total_drift_events": len(all_drifts),
        "selected_features": len(train_df.columns) - 1,
        "infer_lat_ms": infer_lat_ms,
        "train_lat_ms": train_lat_ms,
        "all_drifts": all_drifts,
    }


def save_csv_outputs(warmup: WarmupResult, stream: StreamResult, ev: EvalResult,
                      edge: dict, cfg: Config, out_dir: str = ".") -> None:
    fm = {
        "warmup_accuracy": warmup.warmup_metric.get(), "stream_accuracy": ev.acc,
        "macro_precision": ev.macro_prec, "macro_recall": ev.macro_rec, "macro_f1": ev.macro_f1,
        "weighted_f1": ev.weighted_f1, "detection_rate": ev.det_rate,
        "false_positive_rate": ev.fpr, "binary_f1": ev.bin_f1,
        **{k: v for k, v in edge.items() if k not in ("infer_lat_ms", "train_lat_ms", "all_drifts")},
    }
    pd.DataFrame([fm]).to_csv(f"{out_dir}/final_metrics.csv", index=False)

    rows = []
    for cls in cfg.class_names:
        r = ev.report.get(cls, {})
        rows.append({"class": cls, "precision": r.get("precision", 0), "recall": r.get("recall", 0),
                     "f1_score": r.get("f1-score", 0), "support": r.get("support", 0)})
    per_class_df = pd.DataFrame(rows)
    per_class_df.to_csv(f"{out_dir}/per_class_metrics.csv", index=False)

    pd.DataFrame(ev.cm, index=cfg.class_names, columns=cfg.class_names).to_csv(
        f"{out_dir}/confusion_matrix.csv")

    all_drifts = edge["all_drifts"]
    drift_df = (pd.DataFrame(all_drifts) if all_drifts else
                pd.DataFrame(columns=["packet_index", "phase", "uncertainty",
                                       "replay_samples", "replay_time_ms"]))
    drift_df.to_csv(f"{out_dir}/drift_events.csv", index=False)

    np.save(f"{out_dir}/infer_lat_ms.npy", edge["infer_lat_ms"])
    np.save(f"{out_dir}/train_lat_ms.npy", edge["train_lat_ms"])
    pd.DataFrame(warmup.warmup_acc_log, columns=["idx", "acc"]).to_csv(f"{out_dir}/warmup_acc_log.csv", index=False)
    pd.DataFrame(ev.stream_acc_log, columns=["idx", "acc"]).to_csv(f"{out_dir}/stream_acc_log.csv", index=False)
    pd.DataFrame(stream.uncertainty_log, columns=["idx", "uncertainty"]).to_csv(
        f"{out_dir}/uncertainty_log.csv", index=False)

    logger.info("CSVs + series saved to %s", out_dir)


# ──────────────────────────────────────────────────────────────────────────
# 10. Orchestration
# ──────────────────────────────────────────────────────────────────────────
def run_pipeline(cfg: Config = Config(), out_dir: str = ".") -> dict:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    set_global_seed(cfg)

    train_df, test_df, feature_cols = load_and_split(cfg)
    y_true_test = test_df["Attack_type"].values.tolist()
    n_features = len(feature_cols)

    pipeline = build_model(cfg, train_df, n_features)
    run_compat_gate(pipeline, train_df, feature_cols, cfg)

    det_kswin = KSWIN(window_size=cfg.kswin_window, alpha=cfg.kswin_alpha)
    reservoir = PerClassReservoir(cfg.n_classes, cfg.reservoir_size)

    warmup = run_warmup(pipeline, det_kswin, reservoir, train_df, feature_cols, cfg)
    stream = run_stream(pipeline, det_kswin, reservoir, test_df, feature_cols, cfg,
                         start_peak_mem=warmup.peak_mem_mb, global_start=len(train_df))

    ev = evaluate_post_hoc(y_true_test, stream.y_pred_list, cfg, train_len=len(train_df))
    edge = compute_edge_metrics(pipeline, warmup, stream, train_df, test_df, cfg,
                                 model_path=f"{out_dir}/online_ids_model.pkl")
    save_csv_outputs(warmup, stream, ev, edge, cfg, out_dir=out_dir)

    logger.info("Pipeline complete. Stream accuracy=%.4f Macro-F1=%.4f", ev.acc, ev.macro_f1)
    return {"warmup": warmup, "stream": stream, "eval": ev, "edge": edge}


if __name__ == "__main__":
    run_pipeline()
