#!/usr/bin/env python3
"""
run_baseline_ablation.py
========================
Runs two baseline ablation experiments:

  1. Real Oversampling (champion DNA):
     Duplicate real training images with the same per-class counts as
     each champion's DNA. Isolates whether the value comes from the
     count allocation or from synthetic content.

  2. Class Weighting (inverse frequency):
     Train with no augmentation but weight the loss by inverse class
     frequency. Tests whether simply rebalancing the loss achieves the
     same effect as augmentation.

Usage:
  python run_baseline_ablation.py --config ablation_config.yaml

Requires:
  - train_script_224.py with class_weights support (see patch)
  - Same dependencies as optimize_ldl.py
"""

import argparse
import importlib
import json
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
# Dataset builder: Real Oversampling
# ─────────────────────────────────────────────────────────────────

def build_dataset_real_oversample(
    tmp_root: Path,
    vec: list[float],
    cfg: dict,
    folds: list[int],
):
    """
    Build an augmented dataset by duplicating REAL training images
    with the same per-class counts as the champion DNA.

    Instead of sampling from the fake pool, we sample with replacement
    from the real training images of each class. The images are already
    present in the dataset, so we just add extra entries to NNEW_train.txt.

    Returns: (ds_root, provenance_summary)
    """
    lo, hi = bounds(cfg)
    counts = [max(int(lo), min(int(hi), int(round(v)))) for v in vec]

    ds_root = tmp_root / f"realos-{'-'.join(map(str, counts))}"
    ds_root.mkdir(parents=True, exist_ok=True)

    REAL_ROOT = Path(cfg.get(
        "real_root",
        "/mnt/wd-ssd-4tb/acne_classifiers/datasets/image_restructured_acnescu",
    ))

    real_subdir = cfg.get("real_subdir", None)
    CLASS_IDS = cfg.get("class_ids", [0, 1, 2, 3])
    seed = int(cfg.get("seed", 42))

    rng = np.random.RandomState(seed)

    split_counts = {
        f: {"train_real": 0, "train_oversampled": 0, "val_real": 0, "test_real": 0}
        for f in folds
    }

    for fold in folds:
        fold_out = ds_root / str(fold)
        fold_out.mkdir(parents=True, exist_ok=True)

        real_base = REAL_ROOT / str(fold)

        # ---- Copy all REAL files (train/val/test)
        real_train_by_class = {}  # class -> list of (img, cls, cnt) tuples

        for split in ("train", "val", "test"):
            txt = REAL_ROOT / f"NNEW_{split}_{fold}.txt"
            if not txt.is_file():
                raise FileNotFoundError(f"Missing list: {txt}")

            df_real = pd.read_csv(txt, sep=r"\s+", header=None)

            for _, row in df_real.iterrows():
                img = str(row[0])
                cls = int(row[1])
                cnt = int(row[2]) if row.shape[0] > 2 else 1

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
                    if cls not in real_train_by_class:
                        real_train_by_class[cls] = []
                    real_train_by_class[cls].append((img, cls, cnt))
                elif split == "val":
                    split_counts[fold]["val_real"] += 1
                else:
                    split_counts[fold]["test_real"] += 1

        # ---- Oversample: duplicate real training images per class
        oversample_rows = []

        for cls_idx, (cls, n) in enumerate(zip(CLASS_IDS, counts)):
            if n <= 0:
                continue

            available = real_train_by_class.get(cls, [])
            if not available:
                print(
                    f"[WARN] fold={fold} class={cls}: "
                    f"no real training images to oversample"
                )
                continue

            # Sample WITH replacement
            indices = rng.choice(len(available), size=int(n), replace=True)
            for idx in indices:
                img, c, cnt = available[idx]
                oversample_rows.append((img, c, cnt))
                split_counts[fold]["train_oversampled"] += 1

        # ---- Write NNEW_* files
        real_tr = pd.read_csv(
            REAL_ROOT / f"NNEW_train_{fold}.txt", sep=r"\s+", header=None
        )
        real_va = pd.read_csv(
            REAL_ROOT / f"NNEW_val_{fold}.txt", sep=r"\s+", header=None
        )
        real_te = pd.read_csv(
            REAL_ROOT / f"NNEW_test_{fold}.txt", sep=r"\s+", header=None
        )

        # Append oversampled entries to training list
        if oversample_rows:
            os_df = pd.DataFrame(oversample_rows, columns=[0, 1, 2])
            os_df = os_df.astype({0: str, 1: int, 2: int})
            train_out = pd.concat([real_tr, os_df], ignore_index=True)
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
        "mode": "real_oversample",
        "real_root": str(REAL_ROOT.resolve()),
        "vector_counts": counts,
        "class_ids": CLASS_IDS,
        "folds": folds,
        "per_fold_counts": split_counts,
        "seed": seed,
    }

    return ds_root, prov_summary


# ─────────────────────────────────────────────────────────────────
# Compute class weights from training data
# ─────────────────────────────────────────────────────────────────

def compute_inverse_frequency_weights(
    cfg: dict,
    folds: list[int],
) -> list[float]:
    """
    Compute inverse-frequency class weights from training data.

    weight[c] = n_total / (n_classes * n_c)

    This is the same formula as sklearn's class_weight='balanced'.
    Weights are averaged across folds for consistency.
    """
    REAL_ROOT = Path(cfg.get(
        "real_root",
        "/mnt/wd-ssd-4tb/acne_classifiers/datasets/image_restructured_acnescu",
    ))
    CLASS_IDS = cfg.get("class_ids", [0, 1, 2, 3])
    n_classes = len(CLASS_IDS)

    # Count per-class samples across all folds
    fold_weights = []

    for fold in folds:
        txt = REAL_ROOT / f"NNEW_train_{fold}.txt"
        if not txt.is_file():
            raise FileNotFoundError(f"Missing: {txt}")

        df = pd.read_csv(txt, sep=r"\s+", header=None)
        labels = df[1].astype(int)

        n_total = len(labels)
        class_counts = {c: int((labels == c).sum()) for c in CLASS_IDS}

        weights = []
        for c in CLASS_IDS:
            n_c = class_counts.get(c, 0)
            if n_c == 0:
                weights.append(1.0)
                print(f"[WARN] fold={fold} class={c}: zero samples, weight=1.0")
            else:
                w = n_total / (n_classes * n_c)
                weights.append(w)

        fold_weights.append(weights)

        print(
            f"[CLASS_WEIGHT] fold={fold}: "
            f"counts={[class_counts.get(c, 0) for c in CLASS_IDS]} "
            f"weights={[f'{w:.4f}' for w in weights]}"
        )

    # Average weights across folds
    avg_weights = np.mean(fold_weights, axis=0).tolist()
    print(f"[CLASS_WEIGHT] averaged weights: {[f'{w:.4f}' for w in avg_weights]}")

    return avg_weights


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
    class_weights: list[float] | None = None,
):
    """
    Train the LDL model on a prepared dataset and compute fitness.

    Parameters:
        cfg: full config dict
        run_name: identifier for this run (e.g., "realos_g2_i4")
        ds_path: path to prepared dataset (with NNEW_*.txt files)
        out_dir: output directory for this run
        benchmark: per-fold benchmark thresholds
        folds: list of fold indices
        class_weights: optional class weights for loss weighting.
            IMPORTANT: must be None for real oversampling runs to
            keep variables fixed (only one change at a time).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    tk = cfg.get("trainer_kwargs", {}) or {}

    # Use the class-weighted trainer variant ONLY when class_weights
    # are provided. Otherwise use the original unmodified trainer.
    if class_weights is not None:
        trainer_mod_name = "train_script_224_cw"
        print(f"  [TRAINER] Using class-weighted trainer: {trainer_mod_name}")
    else:
        trainer_mod_name = cfg.get("trainer_module", "train_script_224")
        print(f"  [TRAINER] Using original trainer: {trainer_mod_name}")

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

    # Pass class weights ONLY to the class-weighted trainer
    if class_weights is not None:
        mapped_kwargs["class_weights"] = class_weights

    print(f"\n{'='*60}")
    print(f"  RUNNING: {run_name}")
    print(f"  Dataset: {ds_path}")
    print(f"  Output:  {out_dir}")
    if class_weights is not None:
        print(f"  Class weights: {[f'{w:.4f}' for w in class_weights]}")
    else:
        print(f"  Class weights: None (original loss)")
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
        description="Run baseline ablation experiments: "
        "real oversampling + class weighting"
    )
    ap.add_argument("--config", required=True, help="YAML config file")
    ap.add_argument(
        "--only",
        choices=["real_oversample", "class_weight", "all"],
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

    champions = cfg.get("ablation", {}).get("champions", [])
    results = {}

    # ─── Experiment 1: Real Oversampling ───
    if args.only in ("real_oversample", "all"):
        print("\n" + "=" * 60)
        print("  EXPERIMENT 1: Real Oversampling with Champion DNA")
        print("=" * 60)

        for champ in champions:
            name = champ["name"]
            dna = champ["dna"]
            run_name = f"realos_{name}"
            out_dir = runs_dir / run_name

            if args.skip_existing and (out_dir / "eval_ldl.csv").is_file():
                print(f"  [SKIP] {run_name}: eval_ldl.csv exists")
                continue

            # Build dataset with real oversampling
            tmp_ds_root = Path(tempfile.mkdtemp(dir=tmp_parent, prefix="realos_"))

            try:
                ds_path, prov = build_dataset_real_oversample(
                    tmp_ds_root, dna, cfg, folds
                )

                # Save provenance
                prov_dir = out_dir / "provenance"
                prov_dir.mkdir(parents=True, exist_ok=True)
                (prov_dir / "provenance_summary.json").write_text(
                    json.dumps(prov, indent=2)
                )

                fit = run_single_experiment(
                    cfg, run_name, ds_path, out_dir, benchmark, folds
                )
                results[run_name] = fit

            finally:
                shutil.rmtree(tmp_ds_root, ignore_errors=True)

    # ─── Experiment 2: Class Weighting ───
    if args.only in ("class_weight", "all"):
        print("\n" + "=" * 60)
        print("  EXPERIMENT 2: Class Weighting (Inverse Frequency)")
        print("=" * 60)

        run_name = "class_weight"
        out_dir = runs_dir / run_name

        if args.skip_existing and (out_dir / "eval_ldl.csv").is_file():
            print(f"  [SKIP] {run_name}: eval_ldl.csv exists")
        else:
            # Compute class weights
            weights = compute_inverse_frequency_weights(cfg, folds)

            # Save weights
            prov_dir = out_dir / "provenance"
            prov_dir.mkdir(parents=True, exist_ok=True)
            (prov_dir / "class_weights.json").write_text(
                json.dumps({"weights": weights, "method": "inverse_frequency"}, indent=2)
            )

            # Use the real-only dataset (benchmark) with class weighting
            real_root = Path(cfg["real_root"])

            fit = run_single_experiment(
                cfg, run_name, real_root, out_dir, benchmark, folds,
                class_weights=weights,
            )
            results[run_name] = fit

    # ─── Summary ───
    print("\n" + "=" * 60)
    print("  RESULTS SUMMARY")
    print("=" * 60)

    for name, fit in sorted(results.items()):
        print(f"  {name:<25} fitness = {fit:.6f}")

    # Save summary
    summary_path = exp_root / "ablation_summary.json"
    summary_path.write_text(json.dumps(results, indent=2))
    print(f"\n  Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
