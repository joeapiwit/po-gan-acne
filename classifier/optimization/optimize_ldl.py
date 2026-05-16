#!/usr/bin/env python3
import argparse
import json
import random
import re
import shutil
import tempfile
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

# ───────────────────────── helpers ───────────────────────────────

RUN_ID_RE = re.compile(r"^(g(?P<gen>\d+)_i(?P<idx>\d+))|(bo_(?P<bo>\d+))$")


def folds_from_real_root(real_root: Path) -> list[int]:
    """
    Detect folds from NNEW_train_{fold}.txt under real_root.
    """
    folds = []
    for p in real_root.glob("NNEW_train_*.txt"):
        m = re.search(r"NNEW_train_(\d+)\.txt$", p.name)
        if m:
            folds.append(int(m.group(1)))

    folds = sorted(set(folds))
    if not folds:
        raise RuntimeError(
            f"No folds found under real_root={real_root} "
            f"(expected NNEW_train_*.txt)"
        )
    return folds


def load_state(state_f: Path) -> dict:
    return json.loads(state_f.read_text()) if state_f.is_file() else {}


def save_state(state_f: Path, state: dict) -> None:
    state_f.write_text(json.dumps(state, indent=2))


def clean_incomplete_runs(runs_dir: Path):
    """
    Delete run folders that did not produce a usable eval_ldl.csv.
    Usable = file exists AND has at least 1 non-header row.
    """
    if not runs_dir.exists():
        return

    for p in runs_dir.iterdir():
        if not p.is_dir():
            continue

        f = p / "eval_ldl.csv"
        if (not f.is_file()) or (f.stat().st_size == 0):
            shutil.rmtree(p, ignore_errors=True)
            continue

        try:
            with open(f, "r") as fh:
                lines = [ln.strip() for ln in fh.readlines() if ln.strip()]
            if len(lines) <= 1:
                shutil.rmtree(p, ignore_errors=True)
        except Exception:
            shutil.rmtree(p, ignore_errors=True)


def bounds(cfg):
    lo, hi = cfg.get("count_range", [0, 150])
    return float(lo), float(hi)


def folds_from_benchmark(benchmark_csv: Path) -> list[int]:
    df = pd.read_csv(benchmark_csv)
    folds = sorted(df["fold"].dropna().astype(int).unique().tolist())
    if not folds:
        raise RuntimeError(f"No folds found in {benchmark_csv}")
    return folds


# ───────────────── safe metric selection / legacy repair ───────────────

def cfg_bool(cfg: dict, key: str, default: bool = False) -> bool:
    v = cfg.get(key, default)
    if isinstance(v, bool):
        return v
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "y"}
    return bool(v)


def get_fitness_metric_name(cfg: dict | None = None) -> str:
    """
    Default optimizer metric.

    Final protocol:
      fitness_metric: f1_macro_sane
      allow_legacy_f1_fallback: false

    If an old benchmark CSV lacks f1_macro_sane but has per-class rows,
    the optimizer reconstructs f1_macro_sane in memory.
    """
    cfg = cfg or {}
    return str(cfg.get("fitness_metric", "f1_macro_sane"))


def _safe_num(x, default: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return float(default)

    if not np.isfinite(v):
        return float(default)

    return float(v)


def _parse_class_id(x):
    """
    Parse legacy `class` values into integer class IDs.
    Returns None for macro/average/all rows.
    """
    if pd.isna(x):
        return None

    s = str(x).strip().lower()

    if s in {"macro", "avg", "average", "mean", "all", "total", "ldl", "nan", "none", ""}:
        return None

    try:
        return int(float(s))
    except Exception:
        return None


def numeric_clean_series(s: pd.Series) -> pd.Series:
    """
    Convert metric column to numeric and replace +/-inf with NaN.
    """
    out = pd.to_numeric(s, errors="coerce")
    out = out.replace([np.inf, -np.inf], np.nan)
    return out


def assert_required_eval_columns(df: pd.DataFrame, csv_path: Path) -> None:
    required = {"status", "fold"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{csv_path} missing required columns: {missing}")


def _macro_from_legacy_per_class_rows(
    group: pd.DataFrame,
    metric_col: str,
    class_ids: list[int],
) -> float:
    """
    Compute fixed-label macro metric from legacy per-class rows.

    Rules:
      - fixed class denominator: class_ids, usually [0,1,2,3]
      - missing class metric -> 0.0
      - NaN or inf class metric -> 0.0
      - duplicate rows for same class -> averaged
    """
    vals = []

    for c in class_ids:
        d = group[group["_class_int"] == int(c)]

        if d.empty or metric_col not in d.columns:
            vals.append(0.0)
            continue

        s = numeric_clean_series(d[metric_col]).dropna()
        vals.append(float(s.mean()) if len(s) else 0.0)

    return _safe_num(np.mean(vals), 0.0)


def add_sane_macro_from_legacy_per_class_rows(
    df: pd.DataFrame,
    cfg: dict | None,
    csv_path: Path,
) -> pd.DataFrame:
    """
    Repair old eval_ldl.csv files in memory.

    If f1_macro_sane is absent but the file has per-class rows:
      fold, epoch, status, class, f1

    Then reconstruct:
      f1_macro_sane = mean fixed-class f1 over [0,1,2,3]

    This function does not write to csv_path.
    It only returns a modified DataFrame in memory.
    """
    cfg = cfg or {}

    if "f1_macro_sane" in df.columns:
        return df

    required = {"fold", "epoch", "status", "class", "f1"}
    missing = sorted(required - set(df.columns))

    if missing:
        raise RuntimeError(
            f"{csv_path} missing f1_macro_sane and cannot be repaired from legacy per-class rows.\n"
            f"Missing required columns for repair: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    class_ids = [int(x) for x in cfg.get("class_ids", [0, 1, 2, 3])]

    out = df.copy()

    # Prevent suffix columns if the file was partially modified elsewhere.
    for col in ["f1_macro_sane", "prec_macro_sane", "rec_macro_sane", "specificity_macro_sane"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    out["_class_int"] = out["class"].map(_parse_class_id)
    class_rows = out[out["_class_int"].isin(class_ids)].copy()

    if class_rows.empty:
        raise RuntimeError(
            f"{csv_path} has no usable per-class rows for class_ids={class_ids}. "
            f"Cannot reconstruct f1_macro_sane."
        )

    macro_rows = []

    for keys, g in class_rows.groupby(["fold", "epoch", "status"], dropna=False):
        fold, epoch, status = keys

        row = {
            "fold": fold,
            "epoch": epoch,
            "status": status,
            "f1_macro_sane": _macro_from_legacy_per_class_rows(g, "f1", class_ids),
        }

        if "precision" in g.columns:
            row["prec_macro_sane"] = _macro_from_legacy_per_class_rows(g, "precision", class_ids)

        if "recall" in g.columns:
            row["rec_macro_sane"] = _macro_from_legacy_per_class_rows(g, "recall", class_ids)

        if "specificity" in g.columns:
            row["specificity_macro_sane"] = _macro_from_legacy_per_class_rows(g, "specificity", class_ids)

        macro_rows.append(row)

    macro_df = pd.DataFrame(macro_rows)

    out = out.merge(
        macro_df,
        on=["fold", "epoch", "status"],
        how="left",
    )

    out = out.drop(columns=["_class_int"])

    val = out[out["status"].astype(str) == "val"].copy()

    if val.empty:
        raise RuntimeError(f"{csv_path} has no validation rows after legacy repair.")

    bad = val["f1_macro_sane"].isna().sum()

    if bad > 0:
        raise RuntimeError(
            f"{csv_path} legacy repair failed: "
            f"{bad} validation rows still have NaN f1_macro_sane."
        )

    print(f"[LEGACY_REPAIR] reconstructed f1_macro_sane from per-class rows: {csv_path}")

    return out


def select_score_column(df: pd.DataFrame, cfg: dict | None, csv_path: Path) -> str:
    """
    Select metric column after legacy repair has had a chance to run.
    """
    cfg = cfg or {}
    wanted = get_fitness_metric_name(cfg)
    allow_legacy = cfg_bool(cfg, "allow_legacy_f1_fallback", False)

    if wanted in df.columns:
        return wanted

    # Debug-only escape hatch. Do not use for final paper runs.
    if wanted == "f1_macro_sane" and allow_legacy and "f1" in df.columns:
        warnings.warn(
            f"[LEGACY FALLBACK] {csv_path} has no f1_macro_sane. "
            f"Using legacy f1 because allow_legacy_f1_fallback=true. "
            f"Do not use this for final paper runs.",
            RuntimeWarning,
        )
        return "f1"

    raise RuntimeError(
        f"{csv_path} missing required fitness metric column '{wanted}'.\n"
        f"Available columns: {list(df.columns)}\n"
        f"This should not happen if the file has legacy per-class rows. "
        f"Check columns: fold, epoch, status, class, f1."
    )


def load_benchmark(csv_path: Path, cfg: dict | None = None) -> dict[int, float]:
    """
    Type 1 benchmark mode.

    Preferred:
      benchmark_dir/eval_ldl.csv already has f1_macro_sane.

    Legacy-compatible:
      if f1_macro_sane is missing but the CSV has per-class rows,
      reconstruct f1_macro_sane from fixed class rows in memory.
    """
    cfg = cfg or {}

    df = pd.read_csv(csv_path)
    assert_required_eval_columns(df, csv_path)

    # This is the key fix: old benchmark files are repaired in memory.
    df = add_sane_macro_from_legacy_per_class_rows(df, cfg, csv_path)

    val = df[df["status"].astype(str) == "val"].copy()

    if val.empty:
        raise RuntimeError(f"No validation rows in benchmark CSV: {csv_path}")

    score_col = select_score_column(val, cfg, csv_path)
    val[score_col] = numeric_clean_series(val[score_col])

    benchmark = {}
    bad_folds = []

    for f, grp in val.groupby("fold"):
        f_int = int(f)
        valid_scores = grp[score_col].dropna()

        if valid_scores.empty:
            bad_folds.append(f_int)
            continue

        best = float(valid_scores.max())

        if not np.isfinite(best):
            bad_folds.append(f_int)
            continue

        benchmark[f_int] = best

    if bad_folds:
        raise RuntimeError(
            f"Benchmark CSV has no valid finite '{score_col}' value for folds: {bad_folds}\n"
            f"File: {csv_path}"
        )

    if not benchmark:
        raise RuntimeError(f"No valid benchmark values found in {csv_path}")

    print(f"[BENCHMARK] loaded {len(benchmark)} fold thresholds from {csv_path}")

    for f in sorted(benchmark):
        print(f"[BENCHMARK] fold={f} threshold_{score_col}={benchmark[f]:.6f}")

    return benchmark


def fitness_from_eval(
    eval_csv: Path,
    benchmark: dict[int, float],
    cfg: dict | None = None,
    expected_folds: list[int] | None = None,
    audit_path: Path | None = None,
) -> float:
    """
    Safe benchmark-gated validation fitness.

    For each fold f:
      b_f = benchmark[f]
      best_f = max candidate validation f1_macro_sane for that fold
      gain_f = max(0, best_f - b_f)

    NaN / inf candidate scores are ignored.
    Missing candidate fold gets gain 0.
    """
    cfg = cfg or {}

    df = pd.read_csv(eval_csv)

    if df.empty:
        return float("-inf")

    assert_required_eval_columns(df, eval_csv)

    # Candidate files should already have f1_macro_sane.
    # This also allows old candidate folders to be scored safely if needed.
    df = add_sane_macro_from_legacy_per_class_rows(df, cfg, eval_csv)

    val = df[df["status"].astype(str) == "val"].copy()

    if val.empty:
        return float("-inf")

    score_col = select_score_column(val, cfg, eval_csv)
    val[score_col] = numeric_clean_series(val[score_col])

    if expected_folds is None:
        fold_list = sorted(int(f) for f in benchmark.keys())
    else:
        fold_list = sorted(int(f) for f in expected_folds)

    gains = []

    audit = {
        "eval_csv": str(eval_csv),
        "score_col": score_col,
        "folds": fold_list,
        "per_fold": {},
    }

    for f in fold_list:
        if f not in benchmark:
            raise RuntimeError(
                f"Benchmark missing fold {f}. "
                f"Benchmark folds are {sorted(benchmark.keys())}."
            )

        b = float(benchmark[f])

        if not np.isfinite(b):
            raise RuntimeError(f"Benchmark for fold {f} is not finite: {b}")

        grp = val[val["fold"].astype(int) == int(f)]
        valid_scores = grp[score_col].dropna()

        if valid_scores.empty:
            best = None
            gain = 0.0
            status = "no_valid_candidate_val_score_gain_0"
        else:
            best_val = float(valid_scores.max())

            if not np.isfinite(best_val):
                best = None
                gain = 0.0
                status = "nonfinite_candidate_val_score_gain_0"
            else:
                best = best_val
                gain = max(0.0, best_val - b)
                status = "ok"

        gains.append(float(gain))

        audit["per_fold"][str(f)] = {
            "benchmark": float(b),
            "best_candidate_val": best,
            "gain": float(gain),
            "status": status,
            "valid_candidate_val_rows": int(valid_scores.shape[0]),
            "total_candidate_val_rows": int(grp.shape[0]),
        }

    fitness = float(np.mean(gains)) if gains else float("-inf")
    audit["fitness"] = fitness

    if audit_path is not None:
        audit_path.parent.mkdir(parents=True, exist_ok=True)
        audit_path.write_text(json.dumps(audit, indent=2))

    return fitness

# ───────────── provenance helpers ─────────────────────────────────

def _ensure_subpath(child: Path, root: Path, *, what: str):
    """
    Hard provenance guard: the resolved path must be under the configured root.
    """
    c = child.resolve()
    r = root.resolve()
    try:
        c.relative_to(r)
    except Exception:
        raise RuntimeError(
            f"[PROVENANCE VIOLATION] {what} not under root.\n"
            f"  path={c}\n"
            f"  root={r}"
        )


def _choose_real_src(real_root: Path, real_base: Path, split: str, img: str, prefer_subdir: str | None):
    """
    Robust resolver for REAL images.

    Handles both styles in NNEW_*.txt:
      A) img = "xxx.jpg"
      B) img = "train/xxx.jpg" or val/test path

    Tries with/without explicit split prefix and with/without 'all' and prefer_subdir.
    """
    candidates = []

    if prefer_subdir:
        candidates.append(real_base / split / prefer_subdir / img)
    candidates.append(real_base / split / img)
    candidates.append(real_base / split / "all" / img)

    if prefer_subdir:
        candidates.append(real_base / prefer_subdir / img)
    candidates.append(real_base / img)
    candidates.append(real_base / "all" / img)

    for p in candidates:
        if p.is_file():
            _ensure_subpath(p, real_root, what=f"real image ({split})")
            return p

    return None


# ───────────── dataset building with provenance ───────────────────

def build_dataset(tmp_root: Path, vec: list[float], cfg: dict, folds: list[int]):
    """
    Create augmented dataset under tmp_root and return:
      ds_root, provenance_rows, provenance_summary
    """
    lo, hi = bounds(cfg)
    counts = [max(int(lo), min(int(hi), int(round(v)))) for v in vec]

    ds_root = tmp_root / f"vec-{'-'.join(map(str, counts))}"
    ds_root.mkdir(parents=True, exist_ok=True)

    REAL_ROOT = Path(cfg.get(
        "real_root",
        "/mnt/wd-ssd-4tb/acne_classifiers/datasets/image_restructured_acnescu",
    ))
    FAKE_ROOT = Path(cfg.get(
        "fake_root",
        "/mnt/wd-ssd-4tb/acne_classifiers/datasets/sg2_gen/acnescu_images_fake_50k",
    ))

    real_subdir = cfg.get("real_subdir", None)
    CLASS_IDS = cfg.get("class_ids", [0, 1, 2, 3])
    seed = int(cfg.get("seed", 42))

    rng = np.random.RandomState(seed)
    provenance_rows = []

    split_counts = {
        f: {
            "train_real": 0,
            "train_fake": 0,
            "val_real": 0,
            "test_real": 0,
        }
        for f in folds
    }

    for fold in folds:
        fake_dir = FAKE_ROOT / "data" / str(fold)
        label_txt = FAKE_ROOT / "label" / f"label_fake_{fold}.txt"

        fold_out = ds_root / str(fold)
        fold_out.mkdir(parents=True, exist_ok=True)

        real_base = REAL_ROOT / str(fold)

        # ---- copy REAL files used by train/val/test according to NNEW_*.txt
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
                    REAL_ROOT,
                    real_base,
                    split,
                    img,
                    real_subdir if real_subdir else None,
                )

                if src is None:
                    raise FileNotFoundError(
                        f"Real image not found for fold={fold} split={split} img={img} "
                        f"under {real_base} "
                        f"(tried subdir='{real_subdir}', no-subdir, 'all')."
                    )

                dst = fold_out / img
                dst.parent.mkdir(parents=True, exist_ok=True)

                if not dst.is_file():
                    shutil.copyfile(src, dst)

                provenance_rows.append({
                    "fold": fold,
                    "split": split,
                    "kind": "real",
                    "img": img,
                    "cls": cls,
                    "cnt": cnt,
                    "src_path": str(src.resolve()),
                })

                if split == "train":
                    split_counts[fold]["train_real"] += 1
                elif split == "val":
                    split_counts[fold]["val_real"] += 1
                else:
                    split_counts[fold]["test_real"] += 1

        # ---- select and copy FAKE images for TRAIN only
        df_fake = pd.DataFrame(columns=["img", "cls", "cnt"])

        if label_txt.is_file() and fake_dir.is_dir():
            df_lbl = pd.read_csv(
                label_txt,
                sep=r"\s+",
                header=None,
                names=["img", "cls", "cnt"],
            )

            picks = []

            for cls, n in zip(CLASS_IDS, counts):
                if n <= 0:
                    continue

                df_cls = df_lbl[df_lbl["cls"] == cls]
                if not df_cls.empty:
                    n_take = min(int(n), len(df_cls))
                    picks.append(
                        df_cls.sample(
                            n=n_take,
                            replace=False,
                            random_state=rng,
                        )
                    )

            if picks:
                df_fake = pd.concat(picks).reset_index(drop=True)

            for _, row in df_fake.iterrows():
                img = str(row["img"])
                src = (fake_dir / img).resolve()

                if src.is_file():
                    _ensure_subpath(src, FAKE_ROOT, what="fake image")

                    dst = fold_out / img
                    if not dst.is_file():
                        shutil.copyfile(src, dst)

                    provenance_rows.append({
                        "fold": fold,
                        "split": "train",
                        "kind": "fake",
                        "img": img,
                        "cls": int(row["cls"]),
                        "cnt": int(row["cnt"]),
                        "src_path": str(src),
                    })

                    split_counts[fold]["train_fake"] += 1

        # ---- write new NNEW_* files
        def _write_list(name: str, df: pd.DataFrame):
            (ds_root / f"NNEW_{name}_{fold}.txt").write_text(
                "\n".join(" ".join(map(str, r)) for r in df.values)
            )

        real_tr = pd.read_csv(REAL_ROOT / f"NNEW_train_{fold}.txt", sep=r"\s+", header=None)
        real_va = pd.read_csv(REAL_ROOT / f"NNEW_val_{fold}.txt", sep=r"\s+", header=None)
        real_te = pd.read_csv(REAL_ROOT / f"NNEW_test_{fold}.txt", sep=r"\s+", header=None)

        if not df_fake.empty:
            df_fake_out = df_fake.rename(columns={"img": 0, "cls": 1, "cnt": 2})
            df_fake_out = df_fake_out[[0, 1, 2]].astype({0: str, 1: int, 2: int})
            train_out = pd.concat([real_tr, df_fake_out], ignore_index=True)
        else:
            train_out = real_tr

        _write_list("train", train_out)
        _write_list("val", real_va)
        _write_list("test", real_te)

    prov_summary = {
        "real_root": str(REAL_ROOT.resolve()),
        "fake_root": str(FAKE_ROOT.resolve()),
        "vector_counts": counts,
        "class_ids": CLASS_IDS,
        "folds": folds,
        "per_fold_counts": split_counts,
        "seed": seed,
    }

    return ds_root, provenance_rows, prov_summary


# ───────────── one candidate: build → train → fitness ─────────────

def evaluate_vector(
    vec: list[float],
    cfg: dict,
    run_id: str,
    benchmark: dict[int, float],
    folds: list[int],
    runs_dir: Path,
    tmp_parent: Path,
) -> float:
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))

    out_dir = runs_dir / run_id
    out_dir.mkdir(parents=True, exist_ok=True)

    prov_dir = out_dir / "provenance"
    prov_dir.mkdir(parents=True, exist_ok=True)

    trainer_mod_name = cfg.get("trainer_module", "train_script_224")
    trainer_mod = importlib.import_module(trainer_mod_name)

    if not hasattr(trainer_mod, "run_full_ldl_experiment"):
        raise RuntimeError(
            f"trainer_module '{trainer_mod_name}' has no run_full_ldl_experiment()"
        )

    trainer = getattr(trainer_mod, "run_full_ldl_experiment")

    tk = cfg.get("trainer_kwargs", {}) or {}

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

        save_epoch_csv=bool(tk.get("save_epoch_csvs", tk.get("save_epoch_csv", True))),
        save_epoch_detail_csv=bool(tk.get("save_detail_csvs", tk.get("save_epoch_detail_csv", False))),

        save_weights=bool(tk.get("save_weights", False)),

        eval_splits=tk.get("eval_splits", None),
        append_csv=False,
    )

    tmp_ds_root = Path(tempfile.mkdtemp(dir=tmp_parent, prefix="ds_"))

    try:
        ds_path, prov_rows, prov_summary = build_dataset(tmp_ds_root, vec, cfg, folds)

        pd.DataFrame(prov_rows).to_csv(prov_dir / "provenance_sources.csv", index=False)
        (prov_dir / "provenance_summary.json").write_text(
            json.dumps(prov_summary, indent=2)
        )

        rr = prov_summary["real_root"]
        fr = prov_summary["fake_root"]

        tr = sum(d["train_real"] for d in prov_summary["per_fold_counts"].values())
        tf = sum(d["train_fake"] for d in prov_summary["per_fold_counts"].values())
        vr = sum(d["val_real"] for d in prov_summary["per_fold_counts"].values())
        te = sum(d["test_real"] for d in prov_summary["per_fold_counts"].values())

        print(
            f"[PROVENANCE] {run_id}: "
            f"real_root={rr} | fake_root={fr} | "
            f"train_real={tr} train_fake={tf} val_real={vr} test_real={te}"
        )

        trainer(str(ds_path), str(out_dir), **mapped_kwargs)

        eval_csv = out_dir / "eval_ldl.csv"
        if not eval_csv.is_file():
            raise FileNotFoundError(
                f"Trainer did not write {eval_csv}. "
                f"Optimizer expects this file for fitness_from_eval()."
            )

        fit = fitness_from_eval(
            eval_csv=eval_csv,
            benchmark=benchmark,
            cfg=cfg,
            expected_folds=folds,
            audit_path=out_dir / "fitness_audit.json",
        )

    finally:
        shutil.rmtree(tmp_ds_root, ignore_errors=True)

    return float(fit)


# ───────────────────── genetic algorithm ──────────────────────────

def ga_run_id(gen: int, idx: int) -> str:
    return f"g{gen}_i{idx}"


def genetic_search(
    cfg: dict,
    *,
    resume_state: dict,
    rerun: list[str],
    benchmark: dict[int, float],
    folds: list[int],
    runs_dir: Path,
    tmp_parent: Path,
):
    pop_size = int(cfg["ga"]["pop_size"])
    gens = int(cfg["ga"]["generations"])
    cr = float(cfg["ga"]["crossover_rate"])
    mut_sigma = float(cfg["ga"]["mutation_sigma"])
    lo, hi = bounds(cfg)

    random.seed(cfg.get("seed", 42))
    np.random.seed(cfg.get("seed", 42))

    fitnesses = resume_state.get("fitness", {})
    last_gen = int(resume_state.get("last_gen", -1))
    population = resume_state.get("population", [])

    def _rand_vec():
        return [random.uniform(lo, hi) for _ in range(4)]

    if last_gen < 0 or not population:
        population = [_rand_vec() for _ in range(pop_size)]
        last_gen = -1

    for gen in range(last_gen + 1, gens):
        print(f"\n🧬  GA  generation {gen}/{gens - 1}")

        for idx, vec in enumerate(population):
            rid = ga_run_id(gen, idx)

            if (rid in fitnesses) and (rid not in rerun):
                print(f"  [SKIP] {rid:<8} already evaluated fitness={fitnesses[rid]:.6f}")
                continue

            fit = evaluate_vector(
                vec,
                cfg,
                rid,
                benchmark,
                folds,
                runs_dir,
                tmp_parent,
            )

            fitnesses[rid] = float(fit)
            print(f"  {rid:<8} vec={np.round(vec, 1).tolist()}  fitness={fit:.6f}")

        ranked = sorted(
            [
                (ga_run_id(gen, i), fitnesses[ga_run_id(gen, i)], v)
                for i, v in enumerate(population)
            ],
            key=lambda t: t[1],
            reverse=True,
        )

        elites = [t[2] for t in ranked[:max(1, pop_size // 2)]]

        offspring = []

        while len(offspring) < pop_size - len(elites):
            p1, p2 = random.sample(elites, k=min(2, len(elites)))

            child = []
            for a, b in zip(p1, p2):
                gene = a if random.random() > cr else b
                gene += random.gauss(0.0, mut_sigma)
                gene = max(lo, min(hi, gene))
                child.append(gene)

            offspring.append(child)

        population = elites + offspring

        save_state(
            runs_dir / "ga_state.json",
            {
                "last_gen": gen,
                "population": population,
                "fitness": fitnesses,
            },
        )

    best_run, best_fit = max(fitnesses.items(), key=lambda x: x[1])
    print(f"\n✅  GA finished.  best={best_run}  fitness={best_fit:.6f}")


# ─────────────────── bayesian optimisation ────────────────────────

def bo_run_id(step: int) -> str:
    return f"bo_{step}"


def bayesian_search(
    cfg: dict,
    *,
    resume_state: dict,
    rerun: list[str],
    benchmark: dict[int, float],
    folds: list[int],
    runs_dir: Path,
    tmp_parent: Path,
):
    from skopt import Optimizer

    n_calls = int(cfg["bo"]["n_calls"])
    seed = int(cfg["bo"].get("seed", 42))
    acq = cfg["bo"].get("acq_func", "EI")
    lo, hi = bounds(cfg)

    opt = Optimizer([(lo, hi)] * 4, acq_func=acq, random_state=seed)

    prev_points = resume_state.get("points", [])
    prev_scores = resume_state.get("scores", [])

    for p, s in zip(prev_points, prev_scores):
        opt.tell(p, -s)

    fitnesses = {bo_run_id(i): float(sc) for i, sc in enumerate(prev_scores)}
    step0 = len(prev_points)

    for step in range(step0, n_calls):
        vec = opt.ask()
        rid = bo_run_id(step)

        if rid in fitnesses and rid not in rerun:
            fit = fitnesses[rid]
            print(f"  [SKIP] {rid:<8} already evaluated fitness={fit:.6f}")
        else:
            fit = evaluate_vector(
                vec,
                cfg,
                rid,
                benchmark,
                folds,
                runs_dir,
                tmp_parent,
            )

        opt.tell(vec, -fit)
        fitnesses[rid] = float(fit)

        print(f"  {rid:<8} vec={np.round(vec, 1).tolist()}  fitness={fit:.6f}")

        save_state(
            runs_dir / "bo_state.json",
            {
                "points": opt.Xi,
                "scores": [-fx for fx in opt.yi],
            },
        )

    best_run, best_fit = max(fitnesses.items(), key=lambda x: x[1])
    print(f"\n✅  BO finished.  best={best_run}  fitness={best_fit:.6f}")


# ──────────────────────────── main ────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clean-bad", action="store_true")
    ap.add_argument("--force-run", nargs="+", default=[])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    exp_root = Path(cfg.get(
        "exp_root",
        "/mnt/wd-ssd-4tb/acne_classifiers/experiments/acne04_opt",
    ))

    runs_dir = exp_root / "runs"
    tmp_parent = Path(cfg.get("tmp_parent", "/mnt/wd-ssd-4tb/tmp"))

    exp_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp_parent.mkdir(parents=True, exist_ok=True)

    if args.clean_bad:
        clean_incomplete_runs(runs_dir)

    metric_name = get_fitness_metric_name(cfg)
    allow_legacy = cfg_bool(cfg, "allow_legacy_f1_fallback", False)

    print(f"[FITNESS] metric={metric_name}")
    print(f"[FITNESS] allow_legacy_f1_fallback={allow_legacy}")

    benchmark_dir = cfg.get("benchmark_dir", None)

    if benchmark_dir:
        benchmark_csv = Path(benchmark_dir) / "eval_ldl.csv"
        benchmark = load_benchmark(benchmark_csv, cfg=cfg)
        folds = cfg.get("folds") or folds_from_benchmark(benchmark_csv)
        folds = [int(f) for f in folds]
    else:
        b = float(cfg.get("benchmark_macro_f1", -1.0))
        if not np.isfinite(b):
            raise RuntimeError(f"benchmark_macro_f1 is not finite: {b}")

        real_root = Path(cfg["real_root"])
        folds = cfg.get("folds") or folds_from_real_root(real_root)
        folds = [int(f) for f in folds]

        benchmark = {int(f): b for f in folds}

        print(f"[BENCHMARK] using constant threshold={b:.6f} for folds={folds}")

    algo = cfg.get("algorithm", "ga").lower()
    state_f = runs_dir / f"{algo}_state.json"
    state = load_state(state_f) if args.resume else {}

    if algo == "ga":
        genetic_search(
            cfg,
            resume_state=state,
            rerun=args.force_run,
            benchmark=benchmark,
            folds=folds,
            runs_dir=runs_dir,
            tmp_parent=tmp_parent,
        )
    elif algo == "bo":
        bayesian_search(
            cfg,
            resume_state=state,
            rerun=args.force_run,
            benchmark=benchmark,
            folds=folds,
            runs_dir=runs_dir,
            tmp_parent=tmp_parent,
        )
    else:
        raise RuntimeError(
            f"Unknown algorithm='{algo}'. Expected 'ga' or 'bo'."
        )


if __name__ == "__main__":
    main()