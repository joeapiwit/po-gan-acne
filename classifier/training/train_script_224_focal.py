#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# train_script_224_focal.py
#
# Focal-loss variant of train_script_224_cw.py.
# Adds `loss_mode` parameter:
#   "standard"  - original unweighted KL (identical to train_script_224.py)
#   "weighted"  - per-sample weighted KL using class_weights (same as _cw.py)
#   "focal"     - focal-style weighting: w_i = (1 - p_t)^gamma per sample
#
# When loss_mode="standard" and class_weights=None, behavior is IDENTICAL
# to the original train_script_224.py.
#
# This file exists so the original train_script_224.py is never modified.

import os
import csv
import json
import warnings
import argparse
import time
import math
from pathlib import Path
from datetime import datetime
from typing import Optional, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torchvision import transforms
from torch.utils.data import DataLoader

from dataset import dataset_processing
from utils.report import (
    report_precision_se_sp_yi,
    report_mae_mse,
    report_precision_se_sp_yi_standard_format,
)
from utils.utils import Logger, AverageMeter, time_to_str
from utils.genLD import genLD
from model.resnet50 import resnet50
from transforms.affine_transforms import RandomRotate

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

torch.manual_seed(42)
np.random.seed(42)

BATCH_SIZE = 32
BATCH_SIZE_TEST = 20
LR = 0.001
NUM_WORKERS = 12
WEIGHT_SAVE = False

CLASS_IDS = (0, 1, 2, 3)


# ───────────────────────── path helpers ─────────────────────────

def _peek_first_relpath(txt_file: Path) -> Optional[str]:
    try:
        with open(txt_file, "r") as f:
            for line in f:
                s = line.strip()
                if not s or s.startswith("#"):
                    continue
                tok = s.split()[0]
                if tok:
                    return tok
    except Exception:
        pass
    return None


def _normalize_eval_splits(eval_splits):
    if eval_splits is None:
        return ("train", "val", "test")
    if isinstance(eval_splits, str):
        return (eval_splits,)
    return tuple(eval_splits)


def _resolve_base_for_split(data_root: Path, fold: str, split_txt: Path) -> Path:
    rel = _peek_first_relpath(split_txt)

    if not rel:
        base = data_root / str(fold)
        all_dir = base / "all"
        return all_dir if all_dir.is_dir() else base

    p = Path(rel)

    if p.is_absolute() and p.exists():
        return Path("/")

    for cand in [data_root / str(fold) / "all", data_root / str(fold), data_root]:
        if (cand / rel).exists():
            return cand

    base = data_root / str(fold)
    all_dir = base / "all"
    return all_dir if all_dir.is_dir() else base


# ───────────────────── safe metric functions ─────────────────────

def _safe_float(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)
    if not np.isfinite(v):
        return float(default)
    return float(v)


def _safe_div(num: float, den: float) -> float:
    num = float(num)
    den = float(den)
    if den == 0:
        return 0.0
    out = num / den
    if not np.isfinite(out):
        return 0.0
    return float(out)


def compute_classification_metrics_sane(
    y_true_np: np.ndarray,
    y_pred_np: np.ndarray,
    class_ids=CLASS_IDS,
):
    y_true_np = np.asarray(y_true_np, dtype=int)
    y_pred_np = np.asarray(y_pred_np, dtype=int)

    if y_true_np.shape[0] != y_pred_np.shape[0]:
        raise ValueError(
            f"y_true and y_pred length mismatch: "
            f"{y_true_np.shape[0]} vs {y_pred_np.shape[0]}"
        )

    n = int(y_true_np.shape[0])
    per_class = []

    for cls in class_ids:
        cls = int(cls)
        tp = int(np.sum((y_true_np == cls) & (y_pred_np == cls)))
        fp = int(np.sum((y_true_np != cls) & (y_pred_np == cls)))
        fn = int(np.sum((y_true_np == cls) & (y_pred_np != cls)))
        tn = int(np.sum((y_true_np != cls) & (y_pred_np != cls)))

        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        specificity = _safe_div(tn, tn + fp)

        if precision + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * precision * recall / (precision + recall)
        f1 = _safe_float(f1, 0.0)

        support = int(np.sum(y_true_np == cls))

        per_class.append({
            "class_id": cls,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "support": support,
            "precision": float(precision),
            "recall": float(recall),
            "specificity": float(specificity),
            "f1": float(f1),
        })

    macro_precision = _safe_float(np.mean([r["precision"] for r in per_class]), 0.0)
    macro_recall = _safe_float(np.mean([r["recall"] for r in per_class]), 0.0)
    macro_specificity = _safe_float(np.mean([r["specificity"] for r in per_class]), 0.0)
    macro_f1 = _safe_float(np.mean([r["f1"] for r in per_class]), 0.0)
    acc = _safe_div(int(np.sum(y_true_np == y_pred_np)), n)

    return {
        "precision_macro_sane": macro_precision,
        "recall_macro_sane": macro_recall,
        "specificity_macro_sane": macro_specificity,
        "f1_macro_sane": macro_f1,
        "acc_sane": acc,
        "n_samples": n,
        "per_class": per_class,
    }


def _sanitize_metric_row(row: dict) -> dict:
    out = dict(row)
    for k, v in list(out.items()):
        if isinstance(v, (int, np.integer)):
            out[k] = int(v)
        elif isinstance(v, (float, np.floating)):
            out[k] = _safe_float(v, 0.0)
        else:
            if isinstance(v, str):
                out[k] = v
            else:
                try:
                    fv = float(v)
                    out[k] = _safe_float(fv, 0.0)
                except Exception:
                    out[k] = v
    return out


def _mean_col_safe(df: pd.DataFrame, col: str, default: float = np.nan) -> float:
    if col not in df.columns or len(df) == 0:
        return float(default)
    s = pd.to_numeric(df[col], errors="coerce")
    s = s.replace([np.inf, -np.inf], np.nan)
    if s.dropna().empty:
        return float(default)
    return _safe_float(s.mean(), default)


# ───────────────────── evaluation function ─────────────────────

def evaluation_metrics(model, loader, tag, loss_func, fold, epoch, eval_head: str):
    with torch.no_grad():
        v_loss = 0.0
        y_true_list = []
        y_pred_list = []
        y_pred_m_list = []
        l_true_list = []
        l_pred_list = []

        model.eval()
        records = []

        for step, (x, y, l, paths) in enumerate(loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True)

            cls, cou, c2c = model(x, None)
            loss = loss_func(c2c, y)
            v_loss += float(loss.item())

            if eval_head == "cls":
                logits = cls
            elif eval_head == "c2c":
                logits = c2c
            else:
                logits = cls + c2c

            p = torch.argmax(logits, dim=1)
            pm = torch.argmax(cls + c2c, dim=1)
            pl = torch.argmax(cou, dim=1) + 1

            y_cpu = y.detach().cpu().numpy().astype(int)
            p_cpu = p.detach().cpu().numpy().astype(int)
            pm_cpu = pm.detach().cpu().numpy().astype(int)
            l_cpu = np.asarray(l).astype(int)
            pl_cpu = pl.detach().cpu().numpy().astype(int)

            y_true_list.extend(y_cpu.tolist())
            y_pred_list.extend(p_cpu.tolist())
            y_pred_m_list.extend(pm_cpu.tolist())
            l_true_list.extend(l_cpu.tolist())
            l_pred_list.extend(pl_cpu.tolist())

            for idx, (pr, path) in enumerate(zip(p_cpu, paths)):
                records.append({
                    "epoch": int(epoch),
                    "step": tag,
                    "data_index": int(step * loader.batch_size + idx),
                    "image_path": str(path),
                    "gt": int(y_cpu[idx]),
                    "pred": int(pr),
                })

        v_loss = _safe_float(v_loss / max(1, len(loader)), 0.0)

        y_true = np.asarray(y_true_list, dtype=int)
        y_pred = np.asarray(y_pred_list, dtype=int)
        y_pred_m = np.asarray(y_pred_m_list, dtype=int)
        l_true = np.asarray(l_true_list, dtype=int)
        l_pred = np.asarray(l_pred_list, dtype=int)

        sane = compute_classification_metrics_sane(y_true, y_pred, CLASS_IDS)

        try:
            _ = report_precision_se_sp_yi(y_pred, y_true)
            _ = report_precision_se_sp_yi(y_pred_m, y_true)
            _ = report_mae_mse(l_true, l_pred, y_true)

            out = report_precision_se_sp_yi_standard_format(
                y_pred, y_true, v_loss, fold, epoch, tag, "ldl"
            )

            metrics_list = (
                out.to_dict(orient="records")
                if isinstance(out, pd.DataFrame)
                else list(out)
            )
        except Exception as e:
            print(f"[WARN] legacy report failed for fold={fold} epoch={epoch} tag={tag}: {e}")
            metrics_list = [{}]

        if len(metrics_list) == 0:
            metrics_list = [{}]

        safe_rows = []

        # Build per-class rows from sane metrics (correct per-class values)
        per_class_data = sane["per_class"]  # list of dicts with class_id, precision, recall, f1, specificity

        for cls_idx, cls_metrics in enumerate(per_class_data):
            row = {}
            row["status"] = str(tag)
            row["epoch"] = int(epoch)
            row["fold"] = str(fold)
            row["model"] = "ldl"
            row["model_name"] = "ldl"
            row["class"] = int(cls_metrics["class_id"])
            # Per-class metrics (TRUE per-class, not macro)
            row["precision"] = float(cls_metrics["precision"])
            row["recall"] = float(cls_metrics["recall"])
            row["specificity"] = float(cls_metrics["specificity"])
            row["f1"] = float(cls_metrics["f1"])
            # Macro metrics (same for all class rows)
            row["prec_macro_sane"] = float(sane["precision_macro_sane"])
            row["rec_macro_sane"] = float(sane["recall_macro_sane"])
            row["specificity_macro_sane"] = float(sane["specificity_macro_sane"])
            row["f1_macro_sane"] = float(sane["f1_macro_sane"])
            row["acc_sane"] = float(sane["acc_sane"])
            row["n_samples"] = int(sane["n_samples"])
            row["accuracy"] = float(sane["acc_sane"])
            row["loss"] = float(v_loss)
            safe_rows.append(_sanitize_metric_row(row))

        return safe_rows, records, (y_true, y_pred)


def calculate_val_macro(df: pd.DataFrame, epoch: int, fold: str):
    d = df[
        (df["epoch"] == int(epoch))
        & (df["status"] == "val")
        & (df["fold"].astype(str) == str(fold))
        & (df["model"] == "ldl")
    ]

    if len(d) == 0:
        return (np.nan, np.nan, -np.inf, np.nan, np.inf, np.nan, np.nan, 0)

    prec = _mean_col_safe(d, "prec_macro_sane", np.nan)
    rec = _mean_col_safe(d, "rec_macro_sane", np.nan)
    f1 = _mean_col_safe(d, "f1_macro_sane", -np.inf)
    spec = _mean_col_safe(d, "specificity_macro_sane", np.nan)
    loss = _mean_col_safe(d, "loss", np.nan)
    repo_f1 = _mean_col_safe(d, "f1", np.nan)
    acc = _mean_col_safe(d, "acc_sane", np.nan)

    if "n_samples" in d.columns:
        n = int(pd.to_numeric(d["n_samples"], errors="coerce").fillna(0).mean())
    else:
        n = 0

    return (prec, rec, f1, spec, loss, repo_f1, acc, n)


# ───────────────────── main training function ─────────────────────

def run_full_ldl_experiment(
    data_root: str,
    experiment_root: str,
    *,
    sigma: float = 3.0,
    lam: float = 0.6,
    eval_head: str = "combined",
    img_size: int = 224,
    resize_size: int = 256,
    batch_size: int = 32,
    batch_size_eval: int = 20,
    num_workers: int = 4,
    eval_splits=None,
    max_epochs: int = 120,
    lr: float = 0.001,
    lr_steps=(30, 60, 90),
    patience: int = 20,
    save_weights: bool = False,
    save_epoch_csv: bool = True,
    save_epoch_detail_csv: bool = False,
    images_base: Optional[str] = None,
    folds: Optional[List[str]] = None,
    append_csv: bool = False,
    # ─── NEW: class weighting support ───
    class_weights: Optional[List[float]] = None,
    # ─── NEW: loss mode and focal parameters ───
    loss_mode: str = "standard",
    focal_gamma: float = 2.0,
):
    """
    LDL trainer with optional per-sample class weighting and focal loss.

    loss_mode:
      "standard" - original unweighted KL (class_weights ignored)
      "weighted" - per-sample weighted KL using class_weights
      "focal"    - focal-style: w_i = (1 - p_t)^gamma, class_weights ignored

    When loss_mode="standard" and class_weights=None, behavior is identical
    to train_script_224.py.
    """
    assert eval_head in {"cls", "c2c", "combined"}, "eval_head must be one of {cls,c2c,combined}"
    assert loss_mode in {"standard", "weighted", "focal"}, f"loss_mode must be standard/weighted/focal, got {loss_mode}"

    splits = _normalize_eval_splits(eval_splits)
    data_root = Path(data_root)
    experiment_root = Path(experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    images_base_path = Path(images_base) if images_base else None

    eval_csv_path = experiment_root / "eval_ldl.csv"

    if (not append_csv) and eval_csv_path.is_file():
        eval_csv_path.unlink()

    if not eval_csv_path.is_file():
        eval_csv_path.write_text("")

    log = Logger()
    LOGFILE = experiment_root / f"log_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    log.open(str(LOGFILE), "a")

    auto_folds = (
        sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()])
        or list("01234")
    )
    folds_iter = [str(f) for f in (folds if folds is not None else auto_folds)]

    norm = transforms.Normalize(
        [0.45815152, 0.361242, 0.29348266],
        [0.2814769, 0.226306, 0.20132513],
    )

    T_train = transforms.Compose([
        transforms.Resize((int(resize_size), int(resize_size))),
        transforms.RandomCrop((int(img_size), int(img_size))),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        RandomRotate(20),
        norm,
    ])

    T_eval = transforms.Compose([
        transforms.Resize((int(img_size), int(img_size))),
        transforms.ToTensor(),
        norm,
    ])

    def ds(base_root: Path, txt: Path, aug):
        return dataset_processing.DatasetProcessingWithImagepath(
            base_root,
            str(txt),
            transform=aug,
        )

    def make_loaders(base_root: Path, TR: Path, VA: Path, TE: Path, b_tr: int, b_ev: int):
        ds_tr = ds(base_root, TR, T_train)
        ds_va = ds(base_root, VA, T_eval)
        ds_te = ds(base_root, TE, T_eval)

        nw = int(num_workers)

        L_tr = DataLoader(
            ds_tr, int(b_tr), shuffle=True,
            num_workers=nw, pin_memory=True,
            persistent_workers=(nw > 0),
        )
        L_va = DataLoader(
            ds_va, int(b_ev), shuffle=False,
            num_workers=nw, pin_memory=True,
            persistent_workers=(nw > 0),
        )
        L_te = DataLoader(
            ds_te, int(b_ev), shuffle=False,
            num_workers=nw, pin_memory=True,
            persistent_workers=(nw > 0),
        )
        return L_tr, L_va, L_te

    lr_steps_set = set(
        int(x)
        for x in (
            list(lr_steps)
            if isinstance(lr_steps, (list, tuple))
            else [lr_steps]
        )
    )

    # Log loss mode
    print(f"[LOSS_MODE] {loss_mode}")
    if loss_mode == "focal":
        print(f"[FOCAL] gamma = {focal_gamma}")
    elif loss_mode == "weighted" and class_weights is not None:
        print(f"[CLASS_WEIGHT] Mode: ENABLED, weights = {class_weights}")
    else:
        print(f"[CLASS_WEIGHT] Mode: DISABLED (standard unweighted loss)")

    for fold in folds_iter:
        BATCH_SIZE_TIERS = [int(batch_size), 16, 8, 4, 2, 1]
        BATCH_SIZE_TIERS = [b for b in BATCH_SIZE_TIERS if b > 0]
        tier_idx = 0

        print(f"\n===== START FOLD {fold} =====")

        best_epoch = -1
        best_f1 = -np.inf
        no_imp = 0
        start = time.time()

        fold_root = experiment_root / "ldl" / str(fold)

        for sub in ("weights", "logs", "csv"):
            (fold_root / sub).mkdir(parents=True, exist_ok=True)

        val_hist_path = fold_root / "csv" / "val_history.csv"

        if (not append_csv) and val_hist_path.is_file():
            val_hist_path.unlink()

        if not val_hist_path.is_file():
            val_hist_path.write_text(
                "epoch,f1_macro_sane,acc_sane,prec_macro_sane,"
                "rec_macro_sane,specificity_macro_sane,loss,"
                "bsz_train,bsz_eval\n"
            )

        csv_file = fold_root / "csv" / f"logcsv_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"

        if save_epoch_csv:
            with open(csv_file, "w", newline="") as f:
                csv.writer(f).writerow([
                    "status", "epoch",
                    "loss_cls", "loss_cou", "loss_c2c", "loss_total",
                    "note",
                ])

        TR = data_root / f"NNEW_train_{fold}.txt"
        VA = data_root / f"NNEW_val_{fold}.txt"
        TE = data_root / f"NNEW_test_{fold}.txt"

        base_root = images_base_path if images_base_path else _resolve_base_for_split(data_root, fold, TR)

        print(f"[data] fold {fold}: base_root = {base_root}")

        cur_bsz_tr = BATCH_SIZE_TIERS[tier_idx]
        cur_bsz_ev = int(batch_size_eval) if int(batch_size_eval) > 0 else cur_bsz_tr
        cur_bsz_ev = min(cur_bsz_ev, cur_bsz_tr)

        L_tr, L_va, L_te = make_loaders(base_root, TR, VA, TE, cur_bsz_tr, cur_bsz_ev)

        net = resnet50().cuda()
        cudnn.benchmark = True

        opt = torch.optim.SGD(
            net.parameters(),
            lr=float(lr),
            momentum=0.9,
            weight_decay=5e-4,
        )

        ce = nn.CrossEntropyLoss().cuda()
        kl1 = nn.KLDivLoss().cuda()
        kl2 = nn.KLDivLoss().cuda()
        kl3 = nn.KLDivLoss().cuda()

        # ─── Class weighting: prepare weight tensor ───
        cw_tensor = None
        if loss_mode == "weighted" and class_weights is not None:
            cw_tensor = torch.tensor(class_weights, dtype=torch.float32).cuda()
            print(f"[CLASS_WEIGHT] fold {fold}: weight tensor = {class_weights}")
        elif loss_mode == "focal":
            print(f"[FOCAL] fold {fold}: gamma = {focal_gamma}")

        best_meta = fold_root / "weights" / "best_meta.json"

        for ep in range(int(max_epochs)):
            if ep in lr_steps_set:
                for g in opt.param_groups:
                    g["lr"] *= 0.5

                print(
                    f"LDL fold {fold} | epoch {ep:03d} | "
                    f"LR -> {opt.param_groups[0]['lr']:.6g} | "
                    f"bsz_tr={cur_bsz_tr} bsz_ev={cur_bsz_ev}"
                )

            # ---------------- TRAIN ----------------
            try:
                net.train()
                m_loss = AverageMeter()

                for bx, by, bl, _ in L_tr:
                    bx = bx.cuda(non_blocking=True)
                    bl = bl.numpy() - 1

                    ld = genLD(bl, sigma, "klloss", 65)

                    ld4 = np.vstack((
                        ld[:, :5].sum(1),
                        ld[:, 5:20].sum(1),
                        ld[:, 20:50].sum(1),
                        ld[:, 50:].sum(1),
                    )).T

                    ld = torch.tensor(ld).cuda().float()
                    ld4 = torch.tensor(ld4).cuda().float()

                    cls, cou, c2c = net(bx, None)

                    # ─── Loss computation ───
                    if loss_mode == "focal":
                        # Focal-style weighting: w_i = (1 - p_t)^gamma
                        # p_t = model's predicted probability for true class
                        by_gpu = by.cuda(non_blocking=True)
                        with torch.no_grad():
                            p_t = cls.detach()[
                                torch.arange(cls.size(0), device=cls.device),
                                by_gpu,
                            ]  # (B,)
                            focal_w = (1.0 - p_t).pow(focal_gamma)  # (B,)

                        l_cls = (F.kl_div(torch.log(cls), ld4, reduction='none').sum(1) * focal_w).mean()
                        l_cou = (F.kl_div(torch.log(cou), ld, reduction='none').sum(1) * focal_w).mean()
                        l_c2c = (F.kl_div(torch.log(c2c), ld4, reduction='none').sum(1) * focal_w).mean()

                    elif cw_tensor is not None:
                        # Per-sample weighted KL losses (inverse freq or class-balanced).
                        by_gpu = by.cuda(non_blocking=True)
                        w = cw_tensor[by_gpu]  # (B,) per-sample weight

                        l_cls = (F.kl_div(torch.log(cls), ld4, reduction='none').sum(1) * w).mean()
                        l_cou = (F.kl_div(torch.log(cou), ld, reduction='none').sum(1) * w).mean()
                        l_c2c = (F.kl_div(torch.log(c2c), ld4, reduction='none').sum(1) * w).mean()
                    else:
                        # Original unweighted path (identical to train_script_224.py)
                        l_cls = kl1(torch.log(cls), ld4) * 4
                        l_cou = kl2(torch.log(cou), ld) * 65
                        l_c2c = kl3(torch.log(c2c), ld4) * 4

                    loss = (l_cls + l_c2c) * 0.5 * lam + l_cou * (1 - lam)

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()

                    m_loss.update(float(loss.item()), bx.size(0))

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_tr = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_ev = min(cur_bsz_ev, cur_bsz_tr)

                    print(
                        f"[OOM] TRAIN OOM @ fold={fold} epoch={ep}. "
                        f"fallback bsz_tr={cur_bsz_tr}, bsz_ev={cur_bsz_ev} -> rebuild loaders"
                    )

                    L_tr, L_va, L_te = make_loaders(
                        base_root, TR, VA, TE, cur_bsz_tr, cur_bsz_ev,
                    )
                    continue

                raise

            # ---------------- EVAL ----------------
            fold_metrics = []

            def _maybe_eval(loader, tag: str):
                nonlocal fold_metrics

                met, rec, extra = evaluation_metrics(
                    net, loader, tag, ce, fold, ep, eval_head,
                )

                if save_epoch_csv:
                    pd.DataFrame(met).to_csv(
                        fold_root / "logs" / f"epoch_{ep}_{tag}.csv",
                        index=False,
                    )

                if save_epoch_detail_csv:
                    pd.DataFrame(rec).to_csv(
                        fold_root / "logs" / f"epoch_{ep}_{tag}_detail.csv",
                        index=False,
                    )

                fold_metrics.extend(met)
                return extra

            try:
                y_true_val = None
                y_pred_val = None

                if "train" in splits:
                    _maybe_eval(L_tr, "train")

                y_true_val, y_pred_val = _maybe_eval(L_va, "val")

                if "test" in splits:
                    _maybe_eval(L_te, "test")

            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_tr = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_ev = min(cur_bsz_ev, cur_bsz_tr)

                    print(
                        f"[OOM] EVAL OOM @ fold={fold} epoch={ep}. "
                        f"fallback bsz_tr={cur_bsz_tr}, bsz_ev={cur_bsz_ev} -> rebuild loaders"
                    )

                    L_tr, L_va, L_te = make_loaders(
                        base_root, TR, VA, TE, cur_bsz_tr, cur_bsz_ev,
                    )
                    continue

                raise

            df_epoch = pd.DataFrame(fold_metrics)

            df_epoch.to_csv(
                eval_csv_path,
                mode="a",
                header=(eval_csv_path.stat().st_size == 0),
                index=False,
            )

            prec, rec, f1_macro_sane, spec, val_loss, repo_f1, acc_sane, n_val = calculate_val_macro(
                df_epoch, ep, fold=str(fold),
            )

            with open(val_hist_path, "a") as f:
                f.write(
                    f"{ep},"
                    f"{_safe_float(f1_macro_sane, 0.0):.9f},"
                    f"{_safe_float(acc_sane, 0.0):.9f},"
                    f"{_safe_float(prec, 0.0):.9f},"
                    f"{_safe_float(rec, 0.0):.9f},"
                    f"{_safe_float(spec, 0.0):.9f},"
                    f"{_safe_float(val_loss, 0.0):.9f},"
                    f"{cur_bsz_tr},"
                    f"{cur_bsz_ev}\n"
                )

            improved = bool(f1_macro_sane > best_f1)

            if improved:
                prev = best_f1
                best_f1 = f1_macro_sane
                best_epoch = ep
                no_imp = 0

                if bool(save_weights):
                    torch.save(net.state_dict(), fold_root / "weights" / "best.pth")

                    with open(best_meta, "w") as f:
                        json.dump({
                            "best_epoch": int(best_epoch),
                            "macro_f1_sane": float(best_f1),
                            "val_loss": _safe_float(val_loss, 0.0),
                            "bsz_train": int(cur_bsz_tr),
                            "bsz_eval": int(cur_bsz_ev),
                            "img_size": int(img_size),
                            "resize_size": int(resize_size),
                            "eval_head": eval_head,
                            "class_ids": list(CLASS_IDS),
                            "class_weights": class_weights,
                        }, f, indent=2)

                msg = (
                    f"LDL fold {fold} | NEW BEST @ epoch {ep}: "
                    f"macroF1_sane {prev:.6f} -> {best_f1:.6f}"
                )
                print(msg)
                log.write(msg + "\n")

            else:
                no_imp += 1

            if y_pred_val is None:
                pred_hist = {}
            else:
                unique, counts = np.unique(np.asarray(y_pred_val, dtype=int), return_counts=True)
                pred_hist = dict(zip(unique.tolist(), counts.tolist()))

            diff_repo = (
                repo_f1 - f1_macro_sane
                if np.isfinite(repo_f1) and np.isfinite(f1_macro_sane)
                else np.nan
            )

            cw_tag = " [CW]" if class_weights is not None else ""

            summary = (
                f"LDL fold {fold}{cw_tag} | train {ep:3d} | loss {_safe_float(m_loss.avg, 0.0):.3f} | "
                f"{time_to_str(time.time() - start, 'min')} | "
                f"VAL[n={n_val}] "
                f"acc={_safe_float(acc_sane, 0.0):.6f} "
                f"macroF1={_safe_float(f1_macro_sane, 0.0):.6f} "
                f"(repo_f1={_safe_float(repo_f1, 0.0):.6f}, "
                f"diff={_safe_float(diff_repo, 0.0):.6f}) "
                f"macroPrec={_safe_float(prec, 0.0):.6f} "
                f"macroRec={_safe_float(rec, 0.0):.6f} "
                f"spec={_safe_float(spec, 0.0):.6f} "
                f"loss={_safe_float(val_loss, 0.0):.6f} | "
                f"best_macroF1={_safe_float(best_f1, 0.0):.6f} | "
                f"patience {no_imp}/{patience} | "
                f"pred_hist={pred_hist} | "
                f"bsz_tr={cur_bsz_tr}, bsz_ev={cur_bsz_ev}"
            )

            print(summary)
            log.write(summary + "\n")

            if (not improved) and (no_imp >= int(patience)):
                stop_msg = (
                    f"LDL fold {fold} | EARLY STOP at epoch {ep} "
                    f"(no VAL macroF1 improvement for {no_imp} epochs) | "
                    f"bsz_tr={cur_bsz_tr}, bsz_ev={cur_bsz_ev}"
                )
                print(stop_msg)
                log.write(stop_msg + "\n")
                break

        print(
            f"===== END FOLD {fold} | "
            f"BEST epoch={best_epoch} val_macroF1={_safe_float(best_f1, 0.0):.6f} =====\n"
        )

        log.write(
            f"LDL fold {fold} | BEST epoch={best_epoch} "
            f"macroF1_sane={_safe_float(best_f1, 0.0):.6f}\n"
        )


# ───────────────────────── CLI ─────────────────────────

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--sigma", type=float, default=3.0)
    ap.add_argument("--lam", type=float, default=0.6)
    ap.add_argument("--save_weights", action="store_true")
    ap.add_argument(
        "--eval_head",
        choices=["cls", "c2c", "combined"],
        default="combined",
    )
    ap.add_argument("--images_base", type=str, default=None)
    ap.add_argument("--folds", type=str, default=None)
    ap.add_argument(
        "--class_weights",
        type=float,
        nargs=4,
        default=None,
        help="Per-class loss weights (4 floats). E.g.: --class_weights 1.2 0.8 1.5 2.0",
    )

    args = ap.parse_args()
    WEIGHT_SAVE = args.save_weights

    folds_arg = None
    if args.folds:
        folds_arg = [f.strip() for f in args.folds.split(",") if f.strip() != ""]

    run_full_ldl_experiment(
        args.data,
        args.out,
        sigma=args.sigma,
        lam=args.lam,
        eval_head=args.eval_head,
        images_base=args.images_base,
        folds=folds_arg,
        class_weights=args.class_weights,
    )
