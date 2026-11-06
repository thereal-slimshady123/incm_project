#!/bin/bash
# Plan-scale recovery (Week 2 deliverable) on a single workstation / WSL; same jobs as ada_recovery.sbatch.
#   100 parameter vectors x N in {200,500,1000,2000} for M1, M2, M3, JOINT  + 50 datasets per mechanism for model recovery.
# ~80 CPU-hours (about 5-6 h wall-clock with 13 workers on 20 logical cores / 7 GB RAM; ~300 MB per worker).
# Every finished fit is checkpointed (results/recovery/*.partial.csv) and finished stages are skipped, so the
# script can simply be started again after a suspend or crash.
#
#   bash scripts/local_recovery.sh [n_jobs] 2>&1 | tee -a results/recovery/local_recovery.log
cd "$(dirname "$0")/.." || exit 1
N_JOBS=${1:-13}
mkdir -p results/recovery

if [ ! -f results/recovery/model_recovery_bics.csv ]; then
    python -u run_track_a.py --full-model-recovery --n-datasets 50 --n-jobs "$N_JOBS"
fi
for MODEL in M1 M2 M3 JOINT; do
    if [ ! -f "results/recovery/param_recovery_${MODEL}.csv" ]; then
        python -u run_track_a.py --full-param-recovery --model "$MODEL" \
            --trials 200 500 1000 2000 --n-sims 100 --n-jobs "$N_JOBS"
    fi
done
echo "ALL RECOVERY STAGES DONE"
