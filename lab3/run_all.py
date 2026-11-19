"""Trains every configuration used in lab3.ipynb and caches the results in results/.

usage: python run_all.py [per_series|generic|all] [n_threads] [generic modes, comma-separated]
The notebook calls the same ts_lib.run_experiment(), so running this first only
pre-fills the cache; it is not required.
"""
import sys
import torch
import ts_lib as T

group = sys.argv[1] if len(sys.argv) > 1 else "all"
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 8)
data = {n: T.prepare("../data/L3", n) for n in T.SERIES}

if group in ("per_series", "all"):
    for cal in (False, True):
        for n in T.SERIES:
            T.run_experiment(f"per_{n}_{'cal' if cal else 'y'}", [data[n]], use_cal=cal,
                             stride=1 if n == "TS4011" else 3)

if group in ("generic", "all"):
    modes = sys.argv[3].split(",") if len(sys.argv) > 3 else ["none", "seq", "concat", "init", "hybrid", "film"]
    for mode in modes:
        T.run_experiment(f"generic_{mode}", list(data.values()), use_cal=True, static_mode=mode, stride=4)

if group == "seeds":   # extra seeds of the generic variants to measure seed-to-seed variability
    for seed in (1, 2):
        for mode in (sys.argv[3].split(",") if len(sys.argv) > 3 else ["none", "seq", "concat", "init", "hybrid", "film"]):
            T.run_experiment(f"generic_{mode}_s{seed}", list(data.values()), use_cal=True, static_mode=mode,
                             stride=4, seed=seed)
