#!/bin/bash
# Round 3 on Ada (GPU-accelerated): every step resumes from its saved fits, so resubmitting continues where it stopped.
#   1. model recovery for the robust model class        (scripts/validate_robust_recovery.py, 90 fits)
#   2. fast-guess variant                               (run_c1.py --variant fastguess, 90 fits + rescore)
#   3. replication cohort, robust settings              (10 new mice -> results/c1_replication, 90 fits + rescore)
#   4. M3 with a finer tau_b grid                        (run_c1.py --variant robust_tau, 30 M3 fits x 7 tau values + rescore)
#   5. analysis of the original and the replication cohort
# The diffusion simulations run on the GPUs (INCM_BACKEND=cuda); every fit worker also uses one CPU core for the
# likelihood. If no GPU is usable the job falls back to the CPU and is ~3x slower.
#
#   mkdir -p logs && sbatch scripts/ada_round3.sbatch
#   squeue -u $USER ; tail -f logs/round3_<jobid>.out ; grep -E "STEP|DONE|FAILED" logs/round3_<jobid>.out
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -c 38                  # Ada caps a job at 38 CPUs
#SBATCH --gres=gpu:4
#SBATCH --mem-per-cpu=1500M
#SBATCH --time=2-00:00:00
#SBATCH --job-name=c1_round3
#SBATCH --output=logs/round3_%j.out

set -o pipefail
module load u22/python/3.12.4 2>/dev/null || true    # optional: ~/incm_env carries its own Python
source ~/incm_env/bin/activate
cd ~/incm_project
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1
N_JOBS=${N_JOBS:-${SLURM_CPUS_PER_TASK:-38}}
REP=data/processed/replication

echo "job ${SLURM_JOB_ID:-local} on $(hostname), ${N_JOBS} workers, started $(date)"
nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true
if python -c "from numba import cuda; assert cuda.is_available(); print('CUDA devices:', len(cuda.gpus))"; then
    export INCM_BACKEND=cuda
else
    echo "WARNING: no usable GPU (numba-cuda missing or no driver): running on the CPU"
fi
echo "simulator backend: ${INCM_BACKEND:-numba}"

FAILED=()
step() {                      # step <name> <command...>: run, log, keep going if it fails
    local name=$1; shift
    echo; echo "=== STEP ${name}  $(date)"
    if "$@"; then echo "=== DONE ${name}  $(date)"; else echo "=== FAILED ${name}  $(date)"; FAILED+=("$name"); fi
}

step recovery        python -u scripts/validate_robust_recovery.py --n-jobs "$N_JOBS"
step fastguess_fit   python -u run_c1.py fit --variant fastguess --n-jobs "$N_JOBS"
step fastguess_score python -u run_c1.py rescore --variant fastguess --n-jobs "$N_JOBS"
if [ ! -f "$REP/cohort_rt2p5.parquet" ]; then
    step replication_data python -u run_track_b.py --replication
fi
step replication_fit   python -u run_c1.py fit --variant robust --cohort "$REP/cohort_rt2p5.parquet" --extended "$REP/cohort_extended.parquet" --out results/c1_replication --n-jobs "$N_JOBS"
step replication_score python -u run_c1.py rescore --variant robust --cohort "$REP/cohort_rt2p5.parquet" --extended "$REP/cohort_extended.parquet" --out results/c1_replication --n-jobs "$N_JOBS"
step finetau_fit     python -u run_c1.py fit --variant robust_tau --n-jobs "$N_JOBS"
step finetau_score   python -u run_c1.py rescore --variant robust_tau --n-jobs "$N_JOBS"
step analyze         python -u run_c1.py analyze
step analyze_replication python -u run_c1.py analyze --cohort "$REP/cohort_rt2p5.parquet" --extended "$REP/cohort_extended.parquet" --out results/c1_replication --fig-dir results/figures/c1_replication

if [ ${#FAILED[@]} -eq 0 ]; then
    echo "ALL DONE $(date)"
else
    echo "FINISHED WITH FAILURES: ${FAILED[*]}  $(date)   (fix, then resubmit: finished fits are kept)"
fi
