#!/usr/bin/env python3
# train_script_other_models.py — Standard classifiers (non-LDL) runner
# Models: InceptionResNetV2, ResNet152d (v2-ish), BEiT-L/16-224, EfficientNet-B7, ResNeXt101_32x8d
# Adaptive batch size: try 16 → 8 → 4 on CUDA OOM (train or eval). Retry same epoch after each fallback.

import os, csv, json, warnings, argparse, time
from pathlib import Path
from datetime import datetime
from typing import Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import timm
from timm.data import resolve_data_config, create_transform

from dataset import dataset_processing
from utils.report import (
    report_precision_se_sp_yi,
    report_precision_se_sp_yi_standard_format,
)
from utils.utils import Logger, AverageMeter, time_to_str

# sklearn metrics for guaranteed-correct macro stats
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings("ignore")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
torch.manual_seed(42); np.random.seed(42)

# ─────────────────────────────────────────────────────────────
BATCH_SIZE_TIERS          = [16, 8, 4]  # will try in this order on OOM
LR, NUM_WORKERS           = 0.001, 12
LR_STEPS                  = [30, 60, 90, 120]   # epochs where LR halves
MAX_NO_IMPROVE_EPOCHS     = 20                  # patience on VAL macro-F1 (align LDL)
NUM_CLASSES               = 4
WEIGHT_SAVE               = False               # toggled by --save_weights
# ─────────────────────────────────────────────────────────────

# Friendly names -> timm model names
TIMM_MAP = {
    "inceptionresnetv2":      "inception_resnet_v2",
    "resnet152v2":            "resnet152d",
    "beit-large-patch16-224": "beit_large_patch16_224",
    # IMPORTANT: use TF-converted EfficientNet for pretrained weights in timm
    "efficientnet-b7":        "tf_efficientnet_b7",
    "resnext101_32x8d":       "resnext101_32x8d",
}

# ---------- Robust data-root resolver (handles Acne04 & AcneSCU) ----------
def _peek_first_relpath(txt_file: Path) -> Optional[str]:
    """Return the first non-empty, non-comment first token from a split .txt."""
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

def _resolve_base_for_split(data_root: Path, fold: str, split_txt: Path) -> Path:
    """
    Choose a base directory so that (base / relpath_from_txt) exists.
    Tries, in order:
      1) <root>/<fold>/all
      2) <root>/<fold>
      3) <root>
    If TXT has absolute paths and they exist, return "/" (so "/" / abs == abs).
    """
    rel = _peek_first_relpath(split_txt)
    if not rel:
        base = data_root / str(fold)
        all_dir = base / "all"
        return all_dir if all_dir.is_dir() else base

    p = Path(rel)
    if p.is_absolute() and p.exists():
        return Path("/")

    candidates = [
        data_root / str(fold) / "all",
        data_root / str(fold),
        data_root,
    ]
    for cand in candidates:
        if (cand / rel).exists():
            return cand

    base = data_root / str(fold)
    all_dir = base / "all"
    return all_dir if all_dir.is_dir() else base
# --------------------------------------------------------------------------

def _macro_scores(y_true_np: np.ndarray, y_pred_np: np.ndarray) -> Tuple[float, float, float, float]:
    """Compute macro precision/recall/F1 plus micro-acc for sanity."""
    y_true_np = np.asarray(y_true_np, dtype=int)
    y_pred_np = np.asarray(y_pred_np, dtype=int)
    macro_prec = precision_score(y_true_np, y_pred_np, average="macro", zero_division=0)
    macro_rec  = recall_score(  y_true_np, y_pred_np, average="macro", zero_division=0)
    macro_f1   = f1_score(      y_true_np, y_pred_np, average="macro", zero_division=0)
    acc        = accuracy_score(y_true_np, y_pred_np)
    return macro_prec, macro_rec, macro_f1, acc

def evaluation_metrics_cls(model, loader, tag, loss_func, fold, epoch, device, model_tag):
    """
    Return (metrics_rows: list[dict], record_rows: list[dict], (y_true, y_pred)) for one split.
    Adds 'prec_macro_sane', 'rec_macro_sane', 'f1_macro_sane', 'acc_sane', 'n_samples'.
    """
    with torch.no_grad():
        v_loss = 0.0
        y_true = y_pred = np.array([])
        model.eval(); records = []

        for step, (x, y, _l, paths) in enumerate(loader):  # _l ignored for CLS
            x = x.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
            logits = model(x)
            loss   = loss_func(logits, y)
            v_loss += loss.item()

            p = torch.argmax(logits, dim=1)

            y_true = np.hstack((y_true, y.detach().cpu().numpy()))
            y_pred = np.hstack((y_pred, p.detach().cpu().numpy()))

            for idx, (pr, path) in enumerate(zip(p.cpu(), paths)):
                records.append({
                    "epoch": epoch, "step": tag,
                    "data_index": step * loader.batch_size + idx,
                    "image_path": path, "gt": int(y.cpu()[idx]),
                    "pred": int(pr)
                })

        v_loss /= max(1, len(loader))

        # Keep side-report for parity with repo (not used for control flow)
        _ = report_precision_se_sp_yi(y_pred, y_true)

        # Base table from repo function (gets per-class metrics + loss)
        out = report_precision_se_sp_yi_standard_format(
            y_pred, y_true, v_loss, fold, epoch, tag, model_tag
        )
        if isinstance(out, pd.DataFrame):
            metrics_rows = out.to_dict(orient="records")
        else:
            metrics_rows = list(out)

        # Add guaranteed-correct macro stats
        prec_sane, rec_sane, f1_sane, acc_sane = _macro_scores(y_true, y_pred)
        for row in metrics_rows:
            row["fold"]  = str(fold)
            row["model"] = str(model_tag)
            row["prec_macro_sane"] = float(prec_sane)
            row["rec_macro_sane"]  = float(rec_sane)
            row["f1_macro_sane"]   = float(f1_sane)
            row["acc_sane"]        = float(acc_sane)
            row["n_samples"]       = int(len(y_true))
        return metrics_rows, records, (y_true, y_pred)

def calculate_val_macro(df: pd.DataFrame, epoch: int, fold: str, model_tag: str):
    """
    Pull macro metrics for this fold/epoch on val.
    Prefer explicit sane metrics; include repo 'f1' for visibility.
    """
    d = df[(df["epoch"] == epoch) &
           (df["status"] == "val") &
           (df["fold"] == str(fold)) &
           (df["model"] == str(model_tag))]
    if len(d) == 0:
        return (np.nan, np.nan, -np.inf, np.nan, np.inf, np.nan, np.nan, 0)
    prec = d["prec_macro_sane"].mean() if "prec_macro_sane" in d.columns else d["precision"].mean()
    rec  = d["rec_macro_sane"].mean()  if "rec_macro_sane"  in d.columns else d["recall"].mean()
    f1   = d["f1_macro_sane"].mean()   if "f1_macro_sane"   in d.columns else d["f1"].mean()
    acc  = d["acc_sane"].mean()        if "acc_sane"        in d.columns else np.nan
    spec = d["specificity"].mean()     if "specificity"     in d.columns else np.nan
    loss = d["loss"].mean()            if "loss"            in d.columns else np.nan
    repo_f1 = d["f1"].mean() if "f1" in d.columns else np.nan
    n = int(d["n_samples"].mean()) if "n_samples" in d.columns else 0
    return (prec, rec, f1, spec, loss, repo_f1, acc, n)

# ---- Safe model creation (handles "no pretrained weights" gracefully) ----
def _safe_create_timm(name: str, pretrained: bool, num_classes: int):
    try:
        return timm.create_model(name, pretrained=pretrained, num_classes=num_classes)
    except RuntimeError as e:
        msg = str(e)
        if "No pretrained weights exist" in msg and pretrained:
            print(f"[warn] {name} has no pretrained weights in this timm version; "
                  f"falling back to pretrained=False.")
            return timm.create_model(name, pretrained=False, num_classes=num_classes)
        raise

# ---- Loader factory (so we can rebuild with a smaller batch size on OOM) ----
def _make_loaders(base_root: Path, TR: Path, VA: Path, TE: Path,
                  T_train, T_eval,
                  bsz_train: int, bsz_eval: int, num_workers: int):
    def ds(txt, aug):
        return dataset_processing.DatasetProcessingWithImagepath(
            base_root, str(txt), transform=aug
        )
    ds_train = ds(TR, T_train)
    ds_val   = ds(VA, T_eval)
    ds_test  = ds(TE, T_eval)

    def loader(d, b, sh=False):
        return DataLoader(d, b, shuffle=sh, num_workers=num_workers, pin_memory=True)
    L_tr = loader(ds_train, bsz_train, True)
    L_va = loader(ds_val,   bsz_eval,  False)
    L_te = loader(ds_test,  bsz_eval,  False)
    return L_tr, L_va, L_te

def _clear_cuda():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

def run_full_cls_experiment(data_root: str, experiment_root: str, *,
                            model_name: str, pretrained: bool = True):
    """
    Standard classifiers (non-LDL), aligned with LDL script:
    - Early stopping on VAL macro-F1 (sklearn).
    - Checkpoint on best VAL macro-F1.
    - Evaluate train/val/test every epoch; test never used for selection.
    - Adaptive batch size: 16 → 8 → 4, with epoch retry after fallback.
    """
    data_root = Path(data_root)
    experiment_root = Path(experiment_root)
    experiment_root.mkdir(parents=True, exist_ok=True)

    # Setup logging
    log = Logger()
    LOGFILE = experiment_root / f"log_{model_name}_{datetime.now():%Y-%m-%d_%H-%M-%S}.txt"
    log.open(str(LOGFILE), "a")

    # Numeric folds only (avoid 'all')
    folds = sorted([d.name for d in data_root.iterdir() if d.is_dir() and d.name.isdigit()]) or list("01234")

    # Build a probe model once to derive transforms (safe)
    timm_name = TIMM_MAP[model_name]
    probe = _safe_create_timm(timm_name, pretrained=pretrained, num_classes=NUM_CLASSES)
    data_cfg = resolve_data_config({}, model=probe)
    T_train = create_transform(input_size=data_cfg["input_size"], is_training=True,
                               mean=data_cfg["mean"], std=data_cfg["std"])
    T_eval  = create_transform(input_size=data_cfg["input_size"], is_training=False,
                               mean=data_cfg["mean"], std=data_cfg["std"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cudnn.benchmark = True

    for fold in folds:
        # ───── reset per fold ─────
        print(f"\n===== START FOLD {fold} =====")
        print("Resetting per-fold state: best_f1 = -inf, patience = 0, clearing metrics buffer.")
        best_epoch = -1
        best_macro_f1 = -np.inf
        no_imp = 0
        stuck_count = 0
        last_f1 = None
        all_metrics_rows: List[dict] = []
        start = time.time()

        # ───────── directories ────────────────────────────────────────────
        fold_root = experiment_root / model_name / str(fold)
        for sub in ("weights", "logs", "csv"):
            (fold_root / sub).mkdir(parents=True, exist_ok=True)

        csv_file = fold_root / "csv" / f"logcsv_{datetime.now():%Y-%m-%d_%H-%M-%S}.csv"
        with open(csv_file, "w", newline="") as f:
            csv.writer(f).writerow(
                ["status", "epoch", "loss", "val_loss", "test_loss", "note"]
            )

        # ───────── datasets & loaders ─────────────────────────────────────
        TR = data_root / f"NNEW_train_{fold}.txt"
        VA = data_root / f"NNEW_val_{fold}.txt"
        TE = data_root / f"NNEW_test_{fold}.txt"

        base_root = _resolve_base_for_split(data_root, fold, TR)
        print(f"[data] fold {fold}: base_root = {base_root}")

        # current tier index (0→16, 1→8, 2→4)
        tier_idx = 0
        cur_bsz_train = BATCH_SIZE_TIERS[tier_idx]
        cur_bsz_eval  = BATCH_SIZE_TIERS[tier_idx]

        L_tr, L_va, L_te = _make_loaders(base_root, TR, VA, TE, T_train, T_eval,
                                         cur_bsz_train, cur_bsz_eval, NUM_WORKERS)

        # ───────── model / loss / optimiser ───────────────────────────────
        net = _safe_create_timm(timm_name, pretrained=pretrained, num_classes=NUM_CLASSES).to(device)
        opt = torch.optim.SGD(net.parameters(), lr=LR, momentum=0.9, weight_decay=5e-4)
        ce  = nn.CrossEntropyLoss().to(device)

        best_meta_path = fold_root / "weights" / "best_meta.json"

        ep = 0
        while ep < LR_STEPS[-1]:  # manual loop to allow OOM retry without incrementing epoch
            if ep in LR_STEPS:
                for g in opt.param_groups:
                    g["lr"] *= 0.5
                cur_lr = opt.param_groups[0]["lr"]
                lr_msg = (f"{model_name} fold {fold} | epoch {ep:03d} | "
                          f"LR decayed to {cur_lr:.6g} | "
                          f"bsz_train={cur_bsz_train}, bsz_eval={cur_bsz_eval}")
                print(lr_msg); log.write(lr_msg + "\n")

            # ─── training (OOM-aware) ────────────────────────────────────
            try:
                net.train()
                m_loss = AverageMeter()
                for bx, by, _bl, _ in L_tr:
                    bx = bx.to(device, non_blocking=True); by = by.to(device, non_blocking=True)
                    logits = net(bx)
                    loss = ce(logits, by)
                    opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
                    m_loss.update(loss.item(), bx.size(0))
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_train = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_eval  = BATCH_SIZE_TIERS[tier_idx]
                    print(f"[OOM] TRAIN OOM @ epoch {ep}. Falling back to "
                          f"train={cur_bsz_train}, eval={cur_bsz_eval}. Rebuilding loaders…")
                    log.write(f"[OOM] TRAIN → fallback bsz train={cur_bsz_train}, eval={cur_bsz_eval}\n")
                    L_tr, L_va, L_te = _make_loaders(
                        base_root, TR, VA, TE, T_train, T_eval,
                        cur_bsz_train, cur_bsz_eval, NUM_WORKERS
                    )
                    continue  # retry epoch
                else:
                    raise

            # ─── evaluation (train/val/test) with OOM fallback ───────────
            fold_metrics = []

            # TRAIN metrics (not for early stop)
            try:
                met, rec, _ = evaluation_metrics_cls(net, L_tr, "train", ce, fold, ep, device, model_name)
                pd.DataFrame(met).to_csv(fold_root / "logs" / f"epoch_{ep}_train.csv", index=False)
                pd.DataFrame(rec).to_csv(fold_root / "logs" / f"epoch_{ep}_train_detail.csv", index=False)
                fold_metrics.extend(met)
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_train = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_eval  = BATCH_SIZE_TIERS[tier_idx]
                    print(f"[OOM] EVAL(TRAIN) OOM @ epoch {ep}. Falling back to eval batch size={cur_bsz_eval}.")
                    log.write(f"[OOM] EVAL TRAIN → fallback eval bsz={cur_bsz_eval}\n")
                    L_tr, L_va, L_te = _make_loaders(base_root, TR, VA, TE, T_train, T_eval,
                                                     cur_bsz_train, cur_bsz_eval, NUM_WORKERS)
                    continue
                else:
                    raise

            # VAL (used for early stop)
            try:
                met, rec, (y_true_val, y_pred_val) = evaluation_metrics_cls(net, L_va, "val", ce, fold, ep, device, model_name)
                pd.DataFrame(met).to_csv(fold_root / "logs" / f"epoch_{ep}_val.csv", index=False)
                pd.DataFrame(rec).to_csv(fold_root / "logs" / f"epoch_{ep}_val_detail.csv", index=False)
                fold_metrics.extend(met)
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_train = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_eval  = BATCH_SIZE_TIERS[tier_idx]
                    print(f"[OOM] EVAL(VAL) OOM @ epoch {ep}. Falling back to eval batch size={cur_bsz_eval}.")
                    log.write(f"[OOM] EVAL VAL → fallback eval bsz={cur_bsz_eval}\n")
                    L_tr, L_va, L_te = _make_loaders(base_root, TR, VA, TE, T_train, T_eval,
                                                     cur_bsz_train, cur_bsz_eval, NUM_WORKERS)
                    continue
                else:
                    raise

            # TEST (never used for selection)
            try:
                met, rec, _ = evaluation_metrics_cls(net, L_te, "test", ce, fold, ep, device, model_name)
                pd.DataFrame(met).to_csv(fold_root / "logs" / f"epoch_{ep}_test.csv", index=False)
                pd.DataFrame(rec).to_csv(fold_root / "logs" / f"epoch_{ep}_test_detail.csv", index=False)
                fold_metrics.extend(met)
            except torch.cuda.OutOfMemoryError:
                _clear_cuda()
                if tier_idx < len(BATCH_SIZE_TIERS) - 1:
                    tier_idx += 1
                    cur_bsz_train = BATCH_SIZE_TIERS[tier_idx]
                    cur_bsz_eval  = BATCH_SIZE_TIERS[tier_idx]
                    print(f"[OOM] EVAL(TEST) OOM @ epoch {ep}. Falling back to eval batch size={cur_bsz_eval}.")
                    log.write(f"[OOM] EVAL TEST → fallback eval bsz={cur_bsz_eval}\n")
                    L_tr, L_va, L_te = _make_loaders(base_root, TR, VA, TE, T_train, T_eval,
                                                     cur_bsz_train, cur_bsz_eval, NUM_WORKERS)
                    continue
                else:
                    raise

            all_metrics_rows.extend(fold_metrics)

            # For the simple CSV progress
            val_loss_epoch = float(pd.DataFrame([r for r in fold_metrics if r["status"]=="val"])["loss"].mean())
            test_loss_epoch = float(pd.DataFrame([r for r in fold_metrics if r["status"]=="test"])["loss"].mean())
            with open(csv_file, "a", newline="") as f:
                csv.writer(f).writerow(["val", ep, None, val_loss_epoch, None,
                                        f"bsz_tr={cur_bsz_train},bsz_ev={cur_bsz_eval}"])
                csv.writer(f).writerow(["test", ep, None, None, test_loss_epoch,
                                        f"bsz_tr={cur_bsz_train},bsz_ev={cur_bsz_eval}"])

            # ─── policy: checkpoint & patience on VAL macro-F1 (sane) ────
            df = pd.DataFrame(all_metrics_rows)
            prec, rec, macro_f1_sane, spec, val_loss, repo_f1, acc_sane, n_val = calculate_val_macro(
                df, ep, fold=str(fold), model_tag=model_name
            )

            improved_f1 = macro_f1_sane > best_macro_f1
            if improved_f1:
                prev = best_macro_f1
                best_macro_f1 = macro_f1_sane
                best_epoch = ep
                no_imp = 0
                if WEIGHT_SAVE:
                    (fold_root / "weights").mkdir(parents=True, exist_ok=True)
                    torch.save(net.state_dict(), fold_root / "weights" / "best.pth")
                    with open(best_meta_path, "w") as f:
                        json.dump(
                            {"best_epoch": int(best_epoch),
                             "macro_f1_sane": float(best_macro_f1),
                             "val_loss": float(val_loss),
                             "bsz_train": int(cur_bsz_train),
                             "bsz_eval": int(cur_bsz_eval)}, f
                        )
                print(f"{model_name} fold {fold} | NEW BEST @ epoch {ep}: macroF1_sane {prev:.6f} → {best_macro_f1:.6f}")
                log.write(f"{model_name} fold {fold} | NEW BEST @ epoch {ep}: macroF1_sane {prev:.6f} → {best_macro_f1:.6f}\n")
            else:
                no_imp += 1

            # Sanity: class histogram on val to detect “always one class”
            unique, counts = np.unique(np.asarray(y_pred_val, dtype=int), return_counts=True)
            pred_hist = dict(zip(unique.tolist(), counts.tolist()))
            if last_f1 is not None and abs(macro_f1_sane - last_f1) < 1e-12:
                stuck_count += 1
            else:
                stuck_count = 0
            last_f1 = macro_f1_sane
            if stuck_count >= 3:
                print(f"[warn] fold {fold} epoch {ep}: macroF1_sane unchanged for {stuck_count} epochs. "
                      f"Prediction histogram (val): {pred_hist}")
                log.write(f"[warn] fold {fold} epoch {ep}: macroF1_sane unchanged for {stuck_count} epochs. "
                          f"Prediction histogram (val): {pred_hist}\n")

            # ─── single chatty summary line (aligned with LDL) ───────────
            diff = (repo_f1 - macro_f1_sane) if (not np.isnan(repo_f1)) else np.nan
            summary = (
                f"{model_name} fold {fold} | train {ep:3d} | loss {m_loss.avg:.3f} | {time_to_str(time.time()-start,'min')} | "
                f"VAL[n={n_val}] acc={acc_sane:.6f} macroF1={macro_f1_sane:.6f} "
                f"(repo_f1={repo_f1:.6f}, Δ={diff:.6f}) "
                f"macroPrec={prec:.6f} macroRec={rec:.6f} spec={spec:.6f} "
                f"loss={val_loss:.6f} | best_macroF1={best_macro_f1:.6f}"
                f"{', NEW BEST!' if improved_f1 else ''} | "
                f"patience {no_imp}/{MAX_NO_IMPROVE_EPOCHS} | pred_hist={pred_hist} | "
                f"bsz_tr={cur_bsz_train}, bsz_ev={cur_bsz_eval}"
            )
            print(summary); log.write(summary + "\n")

            # ─── stop condition on F1 patience ───────────────────────────
            if (not improved_f1) and (no_imp >= MAX_NO_IMPROVE_EPOCHS):
                stop_msg = (f"{model_name} fold {fold} | EARLY STOP at epoch {ep} "
                            f"(no VAL macroF1 improvement for {no_imp} epochs) | "
                            f"bsz_tr={cur_bsz_train}, bsz_ev={cur_bsz_eval}")
                print(stop_msg); log.write(stop_msg + "\n")
                break

            ep += 1  # only advance epoch if completed without OOM retries

        print(f"===== END FOLD {fold} | BEST epoch={best_epoch} val_macroF1={best_macro_f1:.6f} =====\n")
        log.write(f"{model_name} fold {fold} | BEST epoch={best_epoch} macroF1_sane={best_macro_f1:.6f}\n")

    print(f"✓ experiment finished for model {model_name}.")
    log.write(f"✓ experiment finished for model {model_name}.\n")

# ───────── CLI ─────────
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Dataset root (acnescu or acne04 root)")
    ap.add_argument("--out",  required=True, help="Output root for logs/weights")
    ap.add_argument("--model", required=True,
                    choices=list(TIMM_MAP.keys()),
                    help="Which classifier to train (non-LDL)")
    ap.add_argument("--no_pretrained", action="store_true")
    ap.add_argument("--save_weights", action="store_true")
    args = ap.parse_args()

    WEIGHT_SAVE = args.save_weights
    Path(args.out).mkdir(parents=True, exist_ok=True)

    run_full_cls_experiment(
        data_root=args.data,
        experiment_root=args.out,
        model_name=args.model,
        pretrained=not args.no_pretrained
    )
