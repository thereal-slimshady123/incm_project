"""
Progress of the C1 fits: how many (mouse, model, seed) fits each variant has finished.

    python scripts/c1_progress.py              # all variants
    python scripts/c1_progress.py robust       # one variant
    python scripts/c1_progress.py robust --out results/c1_replication    # the replication cohort
    watch -n 60 python scripts/c1_progress.py robust

Reads the fit caches in results/c1/ (written after every finished fit), so it works while a job is running
and needs nothing from SLURM. M3 fits (~3x the cost of M1/M2) are dispatched first, so the count rises slowly
at the start and quickly at the end.
"""

import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from run_c1 import DEFAULT_SEEDS, EXTRA_VARIANTS, OUT_DIR, ROBUST_SEEDS, ROBUST_VARIANTS, VARIANT_FILES, rescored_path  # noqa: E402

N_MICE = 10


def expected(variant: str) -> int:
    if variant == "joint":
        return N_MICE * len(DEFAULT_SEEDS)
    if variant == "no_history":
        return N_MICE * 3
    if variant in EXTRA_VARIANTS:
        return N_MICE * len(EXTRA_VARIANTS[variant]["models"]) * len(ROBUST_SEEDS)
    return N_MICE * 3 * len(ROBUST_SEEDS if variant in ROBUST_VARIANTS else DEFAULT_SEEDS)


def main(argv):
    out = OUT_DIR
    if "--out" in argv:
        i = argv.index("--out")
        out, argv = argv[i + 1], argv[:i] + argv[i + 2:]
    variants = argv or list(VARIANT_FILES)
    for v in variants:
        path = os.path.join(out, VARIANT_FILES[v])
        total = expected(v)
        if not os.path.exists(path):
            print(f"{v:12s}  not started ({os.path.relpath(path)} missing)")
            continue
        df = pd.read_csv(path)
        n = len(df)
        line = f"{v:12s}  fits {n:3d}/{total}"
        if n:
            line += "  [" + ", ".join(f"{m} {c}" for m, c in df["model"].value_counts().sort_index().items()) + "]"
        if v in ROBUST_VARIANTS or v in EXTRA_VARIANTS:
            rp = rescored_path(out, v)
            line += f"  rescored {len(pd.read_csv(rp)) if os.path.exists(rp) else 0}/{total}"
        age = (time.time() - os.path.getmtime(path)) / 60
        line += f"  last write {age:.0f} min ago"
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
