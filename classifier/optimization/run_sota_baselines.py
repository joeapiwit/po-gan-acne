#!/usr/bin/env python3
"""
run_sota_baselines.py
=====================
Runs three SOTA baseline experiments for the PO-GAN paper:

  1. Focal Loss (Lin et al. ICCV 2017 adaptation for LDL):
     Train with no augmentation but apply focal-style per-sample
     weighting: w_i = (1 - p_t)^gamma, down-weighting easy examples.

  2. Class-Balanced Loss (Cui et al. CVPR 2019):
     Train with no augmentation but weight the loss by effective-number
     class weights: w_c = (1-beta) / (1-beta^{n_c}).

  3. Zein Proxy (Zein et al. PLOS ONE 2024):
     Build an augmented dataset from our fake pool using Zein's exact
     reported per-class synthetic counts, then train with standard loss.
     Tests whether Zein's hand-picked allocation outperforms the
     optimizer-discovered allocation.

Usage:
  python run_sota_baselines.py --config acne04_sota.yaml
  python run_sota_baselines.py --config acnescu_sota.yaml
  python run_sota_baselines.py --config acne04_sota.yaml --only focal
  python run_sota_baselines.py --config acne04_sota.yaml --only class_balanced
  python run_sota_baselines.py --config acne04_sota.yaml --only zein_proxy

Requires:
  - train_script_224_focal.py  (focal + weighted loss variants)
  - train_script_224.py        (original, for Zein proxy)
  - Same dependencies as optimize_ldl.py
"""

import argparse
import importlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# Import helpers from optimize_ldl.py (must be on sys.path)
from optimize_ldl import (
    bounds,
    folds_from_real_root,
    load_benchmark,
    fitness_from_eval,
    _choose_real_src,
    _ensure_subpath,
)


# ─────────────────────────────────────────────────────────────────
# Class-Balanced weights: Cui et al. CVPR 2019
# ─────────────────────────────────────────────────────────────────

def compute_class_balanced_weights(
    cfg: dict,
    folds: list[int],
    beta: float = 0.9999,
) -> list[float]:
    """
    Compute class-balanced weights using the effective number of samples.

    From Cui et al. "Class-Balanced Loss Based on Effective Number of
    Samples" (CVPR 2019):

        E_c = (1 - beta^{n_c}) / (1 - beta)
        w_c = 1 / E_c  (then normalized so weights sum to n_classes)

    Weights are averaged across folds for consistency.
    """
    REAL_ROOT = Path(cfg["real_root"])
    CLASS_IDS = cfg.get("class_ids", [0, 1, 2, 3])
    n_classes = len(CLASS_IDS)

    fold_weights = []

    for fold in folds:
        txt = REAL_ROOT / f"NNEW_train_{fold}.txt"
        if not txt.is_file():
            raise FileNotFoundError(f"Missing: {txt}")

        df = pd.read_csv(txt, sep=r"\s+", header=None)
        labels = df[1].astype(int)

        class_counts = {c: int((labels == c).sum()) for c in CLASS_IDS}

        # Compute effective number and inverse weights
        raw_weights = []
        for c in CLASS_IDS:
            n_c = class_counts.get(c, 0)
            if n_c == 0:
                raw_weights.append(1.0)
                print(f"[CB_WEIGHT] fold={fold} class={c}: zero samples, weight=1.0")
            else:
                effective_num = (1.0 - beta ** n_c) / (1.0 - beta)
                raw_weights.append(1.0 / effective_num)

        # Normalize so weights sum to n_classes (same scale as uniform weighting)
        w_sum = sum(raw_weights)
        weights = [w * n_classes / w_sum for w in raw_weights]

        fold_weights.append(weights)

        print(
            f"[CB_WEIGHT] fold={fold}: "
            f"counts={[class_counts.get(c, 0) for c in CLASS_IDS]} "
            f"weights={[f'{w:.4f}' for w in weights]}"
        )

    # Average across folds
    avg_weights = np.mean(fold_weights, axis=0).tolist()
    print(f"[CB_WEIGHT] averaged weights: {[f'{w:.4f}' for w in avg_weights]}")

    return avg_weights


# ─────────────────────────────────────────────────────────────────
# Dataset builder: Zein Proxy (fake pool with exact counts)
# ─────────────────────────────────────────────────────────────────

def build_dataset_zein_proxy(
    tmp_root: Path,
    zein_counts: list[int],
    cfg: dict,
    folds: list[int],
):
    """
    Build an augmented dataset by sampling from the fake pool with
    Zein et al.'s exact per-class synthetic counts.

    Structure: real training + val + test images, plus `zein_counts[c]`
    synthetic images per class from `fake_root/data/{c}/`.

    Returns: (ds_root, provenance_summary)
    """
    counts = [int(c) for c in zein_counts]
    tag = f"zein-{'-'.join(map(str, counts))}"

    ds_root = tmp_root / tag
    ds_root.mkdir(parents=True, exist_ok=True)

    REAL_ROOT = Path(cfg["real_root"])
    FAKE_ROOT = Path(cfg["fake_root"])
    real_subdir = cfg.get("real_subdir", None)
    CLASS_IDS = cfg.get("class_ids", [0, 1, 2, 3])
    seed = int(cfg.get("seed", 42))

    rng = np.random.RandomState(seed)

    split_counts = {
        f: {"train_real": 0, "train_fake": 0, "val_real": 0, "test_real": 0}
        for f in folds
    }

    # ---- Discover fake pool files per class ----
    fake_pool = {}
    for c in CLASS_IDS:
        fake_dir = FAKE_ROOT / "data" / str(c)
        if not fake_dir.is_dir():
            # Try without /data/ subdirectory
            fake_dir = FAKE_ROOT / str(c)
        if not fake_dir.is_dir():
            print(f"[WARN] No fake directory for class {c}: {fake_dir}")
            fake_pool[c] = []
            continue
        files = sorted([
            f for f in fake_dir.iterdir()
            if f.is_file() and f.suffix.lower() in (".png", ".jpg", ".jpeg")
        ])
        fake_pool[c] = files
        print(f"[ZEIN] class {c}: {len(files)} fake images available, need {counts[CLASS_IDS.index(c)]}")

    for fold in folds:
        fold_out = ds_root / str(fold)
        fold_out.mkdir(parents=True, exist_ok=True)

        real_base = REAL_ROOT / str(fold)

        # ---- Copy all REAL files (train/val/test) ----
        for split in ("train", "val", "test"):
            txt = REAL_ROOT / f"NNEW_{split}_{fold}.txt"
            if not txt.is_file():
                raise FileNotFoundError(f"Missing list: {txt}")

            df_real = pd.read_csv(txt, sep=r"\s+", header=None)

            for _, row in df_real.iterrows():
                img = str(row[0])
                src = _choose_real_src(
                    REAL_ROOT, real_base, split, img,
                    real_subdir if real_subdir else None,
                )
                if src is None:
                    raise FileNotFoundError(
                        f"Real image not found: fold={fold} split={split} img={img}"
                    )
                dst = fold_out / img
                dst.parent.mkdir(parents=True, exist_ok=True)
                if not dst.is_file():
                    shutil.copyfile(src, dst)

                if split == "train":
                    split_counts[fold]["train_real"] += 1
                elif split == "val":
                    split_counts[fold]["val_real"] += 1
                else:
                    split_counts[fold]["test_real"] += 1

        # ---- Sample fake images per class ----
        fake_rows = []
        for cls_idx, c in enumerate(CLASS_IDS):
            n = counts[cls_idx]
            if n <= 0:
                continue

            pool = fake_pool.get(c, [])
            if len(pool) == 0:
                print(f"[WARN] fold={fold} class={c}: no fake images in pool")
                continue

            if n > len(pool):
                print(
                    f"[WARN] fold={fold} class={c}: requested {n} but only "
                    f"{len(pool)} available, sampling with replacement"
                )
                indices = rng.choice(len(pool), size=n, replace=True)
            else:
                indices = rng.choice(len(pool), size=n, replace=False)

            for idx in indices:
                src_fake = pool[idx]
                # Put fake images in a 'fake_{class}' subdirectory
                rel_name = f"fake_{c}/{src_fake.name}"
                dst_fake = fold_out / rel_name
                dst_fake.parent.mkdir(parents=True, exist_ok=True)
                if not dst_fake.is_file():
                    shutil.copyfile(src_fake, dst_fake)

                # genLD expects lesion count; use midpoint of Hayashi range
                # Class 0: 0 lesions, Class 1: 3, Class 2: 13, Class 3: 30
                hayashi_midpoints = {0: 0, 1: 3, 2: 13, 3: 30}
                cnt = hayashi_midpoints.get(c, 1)
                fake_rows.append((rel_name, c, cnt))
                split_counts[fold]["train_fake"] += 1

        # ---- Write NNEW_* files ----
        real_tr = pd.read_csv(
            REAL_ROOT / f"NNEW_train_{fold}.txt", sep=r"\s+", header=None
        )
        real_va = pd.read_csv(
            REAL_ROOT / f"NNEW_val_{fold}.txt", sep=r"\s+", header=None
        )
        real_te = pd.read_csv(
            REAL_ROOT / f"NNEW_test_{fold}.txt", sep=r"\s+", header=None
        )

        # Append fake entries to training list
        if fake_rows:
            fake_df = pd.DataFrame(fake_rows, columns=[0, 1, 2])
            fake_df = fake_df.astype({0: str, 1: int, 2: int})
            train_out = pd.concat([real_tr, fake_df], ignore_index=True)
        else:
            train_out = real_tr

        def _write_list(name, df):
            (ds_root / f"NNEW_{name}_{fold}.txt").write_text(
                "\n".join(" ".join(map(str, r)) for r in df.values)
            )

        _write_list("train", train_out)
        _write_list("val", real_va)
        _write_list("test", real_te)

    prov_summary = {
        "mode": "zein_proxy",
        "real_root": str(REAL_ROOT.resolve()),
        "fake_root": str(FAKE_ROOT.resolve()),
        "zein_counts": counts,
        "class_ids": CLASS_IDS,
        "folds": folds,
        "per_fold_counts": split_counts,
        "seed": seed,
        "source": "Zein et al. PLOS ONE 2024, Table: 1387/841/1173/1086",
    }

    return ds_root, prov_summary


# ─────────────────────────────────────────────────────────────────
# Run a single experiment
# ─────────────────────────────────────────────────────────────────

def run_single_experiment(
    cfg: dict,
    run_name: str,
    ds_path: Path,
    out_dir: Path,
    benchmark: dict[int, float],
    folds: list[int],
    trainer_mod_name: str = "train_script_224",
    class_weights: list[float] | None = None,
    loss_mode: str = "standard",
    focal_gamma: float = 2.0,
):
    """
    Train the LDL model on a prepared dataset and compute fitness.

    Parameters:
        cfg:              full config dict
        run_name:         identifier for this run
        ds_path:          path to prepared dataset (with NNEW_*.txt files)
        out_dir:          output directory for this run
        benchmark:        per-fold benchmark thresholds
        folds:            list of fold indices
        trainer_mod_name: which trainer module to import
        class_weights:    optional class weights for loss weighting
        loss_mode:        "standard", "weighted", or "focal"
        focal_gamma:      gamma for focal loss (only if loss_mode="focal")
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tk = cfg.get("trainer_kwargs", {}) or {}

    trainer_mod = importlib.import_module(trainer_mod_name)
    trainer = getattr(trainer_mod, "run_full_ldl_experiment")

    mapped_kwargs = dict(
        sigma=float(tk.get("sigma", 3.0)),
        lam=float(tk.get("lam", 0.6)),
        eval_head=str(tk.get("eval_head", "combined")),
        img_size=int(tk.get("img_size", 224)),
        resize_size=int(tk.get("resize_size", 256)),
        batch_size=int(tk.get("train_bs", tk.get("batch_size", 32))),
        batch_size_eval=int(tk.get("eval_bs", tk.get("batch_size_eval", 20))),
        num_workers=int(tk.get("num_workers", 4)),
        lr=float(tk.get("lr", 0.001)),
        lr_steps=tuple(tk.get("lr_steps", (30, 60, 90))),
        patience=int(tk.get("patience", 20)),
        save_epoch_csv=bool(tk.get("save_epoch_csvs", True)),
        save_epoch_detail_csv=bool(tk.get("save_detail_csvs", False)),
        save_weights=bool(tk.get("save_weights", False)),
        eval_splits=tk.get("eval_splits", None),
        append_csv=False,
    )

    # Pass loss variant parameters only to the focal trainer
    if trainer_mod_name == "train_script_224_focal":
        mapped_kwargs["loss_mode"] = loss_mode
        mapped_kwargs["focal_gamma"] = focal_gamma
        if class_weights is not None:
            mapped_kwargs["class_weights"] = class_weights
    elif class_weights is not None:
        # For the cw trainer
        mapped_kwargs["class_weights"] = class_weights

    print(f"\n{'='*60}")
    print(f"  RUNNING: {run_name}")
    print(f"  Dataset: {ds_path}")
    print(f"  Output:  {out_dir}")
    print(f"  Trainer: {trainer_mod_name}")
    print(f"  Loss mode: {loss_mode}")
    if loss_mode == "focal":
        print(f"  Focal gamma: {focal_gamma}")
    if class_weights is not None:
        print(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")
    print(f"{'='*60}\n")

    trainer(str(ds_path), str(out_dir), **mapped_kwargs)

    eval_csv = out_dir / "eval_ldl.csv"
    if not eval_csv.is_file():
        raise FileNotFoundError(f"Trainer did not write {eval_csv}")

    fit = fitness_from_eval(
        eval_csv=eval_csv,
        benchmark=benchmark,
        cfg=cfg,
        expected_folds=folds,
        audit_path=out_dir / "fitness_audit.json",
    )

    print(f"\n  {run_name}: fitness = {fit:.6f}")
    return fit


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Run SOTA baseline experiments: "
        "Focal Loss, Class-Balanced Loss, Zein Proxy"
    )
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument(
        "--only",
        choices=["focal", "class_balanced", "zein_proxy", "all"],
        default="all",
        help="Which experiment(s) to run",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip runs that already have eval_ldl.csv",
    )
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    exp_root = Path(cfg["exp_root"])
    runs_dir = exp_root / "runs"
    tmp_parent = Path(cfg.get("tmp_parent", "/mnt/wd-ssd-4tb/tmp"))

    exp_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp_parent.mkdir(parents=True, exist_ok=True)

    # Load benchmark
    benchmark_dir = cfg.get("benchmark_dir", None)
    if benchmark_dir:
        benchmark_csv = Path(benchmark_dir) / "eval_ldl.csv"
        benchmark = load_benchmark(benchmark_csv, cfg=cfg)
        folds = cfg.get("folds") or [
            int(f)
            for f in sorted(
                pd.read_csv(benchmark_csv)["fold"]
                .dropna()
                .astype(int)
                .unique()
            )
        ]
    else:
        real_root = Path(cfg["real_root"])
        folds = cfg.get("folds") or folds_from_real_root(real_root)
        b = float(cfg["benchmark_macro_f1"])
        benchmark = {int(f): b for f in folds}

    folds = [int(f) for f in folds]

    results = {}

    # ═══════════════════════════════════════════════════════════════
    # Experiment 1: Focal Loss
    # ═══════════════════════════════════════════════════════════════
    if args.only in ("focal", "all"):
        print("\n" + "=" * 60)
        print("  EXPERIMENT 1: Focal Loss (gamma=2.0)")
        print("=" * 60)

        focal_cfg = cfg.get("focal", {})
        gamma = float(focal_cfg.get("gamma", 2.0))

        run_name = f"focal_g{gamma:.1f}"
        out_dir = runs_dir / run_name

        if args.skip_existing and (out_dir / "eval_ldl.csv").is_file():
            print(f"  [SKIP] {run_name}: eval_ldl.csv exists")
        else:
            # Save config
            prov_dir = out_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            (prov_dir / "focal_config.json").write_text(
                json.dumps({"loss_mode": "focal", "gamma": gamma}, indent=2)
            )

            # Train on real-only data with focal loss
            real_root = Path(cfg["real_root"])

            fit = run_single_experiment(
                cfg, run_name, real_root, out_dir, benchmark, folds,
                trainer_mod_name="train_script_224_focal",
                loss_mode="focal",
                focal_gamma=gamma,
            )
            results[run_name] = fit

    # ═══════════════════════════════════════════════════════════════
    # Experiment 2: Class-Balanced Loss (Cui et al. CVPR 2019)
    # ═══════════════════════════════════════════════════════════════
    if args.only in ("class_balanced", "all"):
        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: Class-Balanced Loss (Cui et al.)")
        print("=" * 60)

        cb_cfg = cfg.get("class_balanced", {})
        beta = float(cb_cfg.get("beta", 0.9999))

        run_name = "class_balanced"
        out_dir = runs_dir / run_name

        if args.skip_existing and (out_dir / "eval_ldl.csv").is_file():
            print(f"  [SKIP] {run_name}: eval_ldl.csv exists")
        else:
            # Compute class-balanced weights
            cb_weights = compute_class_balanced_weights(cfg, folds, beta=beta)

            # Save weights
            prov_dir = out_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            (prov_dir / "class_balanced_weights.json").write_text(
                json.dumps({
                    "method": "class_balanced_effective_number",
                    "beta": beta,
                    "weights": cb_weights,
                    "reference": "Cui et al. CVPR 2019",
                }, indent=2)
            )

            # Train on real-only data with class-balanced weights
            # Use the focal trainer in "weighted" mode (same as cw trainer
            # but we keep one trainer file for all variants)
            real_root = Path(cfg["real_root"])

            fit = run_single_experiment(
                cfg, run_name, real_root, out_dir, benchmark, folds,
                trainer_mod_name="train_script_224_focal",
                class_weights=cb_weights,
                loss_mode="weighted",
            )
            results[run_name] = fit

    # ═══════════════════════════════════════════════════════════════
    # Experiment 3: Zein Proxy
    # ═══════════════════════════════════════════════════════════════
    if args.only in ("zein_proxy", "all"):
        zein_cfg = cfg.get("zein_proxy", {})

        if not zein_cfg.get("enabled", True):
            print("\n[SKIP] Zein proxy: disabled in config")
        else:
            print("\n" + "=" * 60)
            print("  EXPERIMENT 3: Zein Proxy (exact reported counts)")
            print("=" * 60)

            zein_counts = zein_cfg["counts"]

            run_name = "zein_proxy"
            out_dir = runs_dir / run_name

            if args.skip_existing and (out_dir / "eval_ldl.csv").is_file():
                print(f"  [SKIP] {run_name}: eval_ldl.csv exists")
            else:
                # Build dataset with Zein's counts from fake pool
                tmp_ds_root = Path(
                    tempfile.mkdtemp(dir=tmp_parent, prefix="zein_proxy_")
                )

                try:
                    ds_path, prov = build_dataset_zein_proxy(
                        tmp_ds_root, zein_counts, cfg, folds
                    )

                    # Save provenance
                    prov_dir = out_dir / "provenance"
                    prov_dir.mkdir(parents=True, exist_ok=True)
                    (prov_dir / "provenance_summary.json").write_text(
                        json.dumps(prov, indent=2)
                    )

                    # Train with standard (unweighted) loss on
                    # augmented dataset.
                    # Use train_script_224_focal with loss_mode="standard"
                    # so eval_ldl.csv gets true per-class metrics.
                    fit = run_single_experiment(
                        cfg, run_name, ds_path, out_dir, benchmark, folds,
                        trainer_mod_name="train_script_224_focal",
                        loss_mode="standard",
                    )
                    results[run_name] = fit

                finally:
                    shutil.rmtree(tmp_ds_root, ignore_errors=True)

    # ═══════════════════════════════════════════════════════════════
    # Summary
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for name, fit in sorted(results.items()):
        print(f"  {name:<25} fitness = {fit:.6f}")

    # Save summary
    summary_path = exp_root / "sota_baselines_summary.json"

    # Merge with existing summary if present
    if summary_path.is_file():
        existing = json.loads(summary_path.read_text())
        existing.update(results)
        results = existing

    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
