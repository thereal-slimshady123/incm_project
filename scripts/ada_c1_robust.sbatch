#!/bin/bash
# Robust C1 (v2) on Ada: fit -> common-seed re-scoring -> analysis, for ONE variant.  CPU only (numba simulator).
#
#   VARIANT=robust : wide bounds + lapse/contaminant mixture + inferred (ideal-observer) prior   <- the main run
#   VARIANT=lapse  : wide bounds + lapse/contaminant mixture, true block prior                  <- ablation
#
# Submit (from ~/incm_project on the Ada head node):
#   mkdir -p logs
#   sbatch --export=ALL,VARIANT=robust scripts/ada_c1_robust.sbatch
#   sbatch --export=ALL,VARIANT=lapse --dependency=afterany:<robust job id> scripts/ada_c1_robust.sbatch
#
# Watch:   squeue -u $USER          tail -f logs/c1_<jobid>.out          python scripts/c1_progress.py
# Resume:  every finished fit is saved immediately; if the job dies or hits its time limit, submit the same
#          command again and it continues where it stopped.
# One-time setup, copying the repo and fetching results: README section "Robust C1 (v2) on Ada".
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -c 38                  # Ada caps a job at 38 CPUs
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=1-00:00:00
#SBATCH --job-name=c1_robust
#SBATCH --output=logs/c1_%j.out

set -eo pipefail      # no -u: `module load` and venv activation can read unset variables
VARIANT=${VARIANT:-robust}
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-38}}

module load u22/python/3.12.4 2>/dev/null || true    # optional: ~/incm_env carries its own Python
source ~/incm_env/bin/activate
cd ~/incm_project

# one worker per core: stop numpy / BLAS from starting extra threads inside each worker
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1

echo "job ${SLURM_JOB_ID:-local} on $(hostname), variant=${VARIANT}, ${N_JOBS} workers, started $(date)"
python -u run_c1.py fit --variant "$VARIANT" --n-jobs "$N_JOBS"
echo "fits done $(date)"
python -u run_c1.py rescore --variant "$VARIANT" --n-jobs "$N_JOBS"
echo "rescore done $(date)"
python -u run_c1.py analyze
echo "ALL DONE $(date)  ->  results/c1/c1_report.md, results/c1/c1_${VARIANT}_*, results/figures/c1/"
