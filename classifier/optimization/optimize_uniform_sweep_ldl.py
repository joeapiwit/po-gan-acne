#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import pandas as pd
import yaml

from optimize_ldl import (
    clean_incomplete_runs,
    evaluate_vector,
    fitness_from_eval,
    folds_from_benchmark,
    folds_from_real_root,
    load_benchmark,
)


def resolve_benchmark_and_folds(cfg: dict):
    benchmark_dir = cfg.get("benchmark_dir", None)

    if benchmark_dir:
        benchmark_csv = Path(benchmark_dir) / "eval_ldl.csv"
        benchmark = load_benchmark(benchmark_csv)
        folds = cfg.get("folds") or folds_from_benchmark(benchmark_csv)
    else:
        b = float(cfg.get("benchmark_macro_f1", -1.0))
        real_root = Path(cfg["real_root"])
        folds = cfg.get("folds") or folds_from_real_root(real_root)
        benchmark = {int(f): b for f in folds}

    folds = [int(f) for f in folds]
    return benchmark, folds


def build_uniform_values(cfg: dict) -> list[int]:
    ug = cfg.get("uniform_grid", {}) or {}
    start = int(ug.get("start", 10))
    stop = int(ug.get("stop", 150))
    step = int(ug.get("step", 10))

    if step <= 0:
        raise ValueError("uniform_grid.step must be > 0")
    if start > stop:
        raise ValueError("uniform_grid.start must be <= uniform_grid.stop")

    vals = list(range(start, stop + 1, step))
    if not vals:
        raise ValueError("Uniform grid is empty")

    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--clean-bad", action="store_true")
    ap.add_argument("--force-run", nargs="+", default=[])
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())

    exp_root = Path(cfg["exp_root"])
    runs_dir = exp_root / "runs"
    tmp_parent = Path(cfg.get("tmp_parent", "/mnt/wd-ssd-4tb/tmp"))

    exp_root.mkdir(parents=True, exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)
    tmp_parent.mkdir(parents=True, exist_ok=True)

    if args.clean_bad:
        clean_incomplete_runs(runs_dir)

    benchmark, folds = resolve_benchmark_and_folds(cfg)

    values = build_uniform_values(cfg)
    prefix = str((cfg.get("uniform_grid", {}) or {}).get("run_prefix", "uniform"))

    summary_rows = []
    summary_csv = exp_root / "uniform_summary.csv"
    summary_json = exp_root / "uniform_summary.json"

    if args.resume and summary_csv.is_file():
        old = pd.read_csv(summary_csv)
        if not old.empty:
            summary_rows.extend(old.to_dict(orient="records"))

    done_run_ids = {str(r["run_id"]) for r in summary_rows if "run_id" in r}

    print(f"[UNIFORM SWEEP] values = {values}")
    print(f"[UNIFORM SWEEP] total runs = {len(values)}")

    for n in values:
        vec = [float(n), float(n), float(n), float(n)]
        run_id = f"{prefix}_{n:03d}"
        out_dir = runs_dir / run_id
        eval_csv = out_dir / "eval_ldl.csv"

        if args.resume and run_id in done_run_ids and run_id not in args.force_run:
            print(f"[SKIP] {run_id} already in summary")
            continue

        if args.resume and eval_csv.is_file() and run_id not in args.force_run:
            fit = float(fitness_from_eval(eval_csv, benchmark))
            print(f"[RESUME-USE] {run_id} vec={vec} fitness={fit:.6f}")
        else:
            fit = float(
                evaluate_vector(
                    vec=vec,
                    cfg=cfg,
                    run_id=run_id,
                    benchmark=benchmark,
                    folds=folds,
                    runs_dir=runs_dir,
                    tmp_parent=tmp_parent,
                )
            )
            print(f"[DONE] {run_id} vec={vec} fitness={fit:.6f}")

        row = {
            "run_id": run_id,
            "uniform_count": int(n),
            "c0": int(n),
            "c1": int(n),
            "c2": int(n),
            "c3": int(n),
            "fitness": float(fit),
            "eval_csv": str(eval_csv),
        }

        summary_rows = [r for r in summary_rows if str(r.get("run_id")) != run_id]
        summary_rows.append(row)

        df = pd.DataFrame(summary_rows).sort_values(["uniform_count", "run_id"]).reset_index(drop=True)
        df.to_csv(summary_csv, index=False)
        summary_json.write_text(json.dumps(df.to_dict(orient="records"), indent=2))

    if not summary_rows:
        print("[DONE] no runs executed")
        return

    df = pd.DataFrame(summary_rows).sort_values("fitness", ascending=False).reset_index(drop=True)
    best = df.iloc[0].to_dict()

    print("\n===== UNIFORM SWEEP FINISHED =====")
    print(f"best_run   : {best['run_id']}")
    print(f"best_count : {int(best['uniform_count'])}")
    print(f"best_vec   : [{int(best['c0'])}, {int(best['c1'])}, {int(best['c2'])}, {int(best['c3'])}]")
    print(f"best_fit   : {float(best['fitness']):.6f}")
    print(f"summary_csv: {summary_csv}")


if __name__ == "__main__":
    main()