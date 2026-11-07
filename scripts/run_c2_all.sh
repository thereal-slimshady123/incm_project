#!/usr/bin/env bash
# Whole Week 3 / C2 computation on one workstation (resumable: every finished fit is cached in results/c2/*.csv).
#   bash scripts/run_c2_all.sh 12        # 12 parallel workers (each ~350 MB of RAM)
# Order: bound-type recovery check (so the result can be read with its error rates), the main fits, the collapse profile, the analysis.
set -euo pipefail
cd "$(dirname "$0")/.."
N="${1:-12}"
mkdir -p results/c2
python scripts/validate_c2_recovery.py --n-datasets 4 --n-jobs "$N" 2>&1 | tee -a results/c2/validation/validation_log.txt
python run_c2.py fit --n-jobs "$N" 2>&1 | tee -a results/c2/fit_log.txt
python scripts/profile_collapse.py --rt-max 5.0 --n-jobs "$N" 2>&1 | tee -a results/c2/profile_log.txt
python run_c2.py analyze 2>&1 | tee -a results/c2/analyze_log.txt
