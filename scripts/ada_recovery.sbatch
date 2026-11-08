#!/bin/bash
# Track A plan-scale recovery on Ada (CPU only: the numba simulator does not need a GPU).
#
# One-time setup on the Ada head node (IIIT VPN required off campus):
#   module load u22/python/3.12.4
#   python3 -m venv ~/incm_env && source ~/incm_env/bin/activate
#   pip install -r ~/incm_project/requirements.txt
#
# Copy the repo from your laptop:   scp -r incm_project <user>@ada-gw1.iiit.ac.in:
# Submit:                           cd ~/incm_project && sbatch scripts/ada_recovery.sbatch
# Watch:                            tail -f recovery_%j.out
# Fetch results:                    scp -r <user>@ada-gw1.iiit.ac.in:incm_project/results/recovery .
#
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -c 40
#SBATCH --mem-per-cpu=2G
#SBATCH --time=1-00:00:00
#SBATCH --output=recovery_%j.out

module load u22/python/3.12.4
source ~/incm_env/bin/activate
cd ~/incm_project

N_JOBS=${SLURM_CPUS_PER_TASK:-40}

for MODEL in M1 M2 M3 JOINT; do
    python -u run_track_a.py --full-param-recovery --model "$MODEL" \
        --trials 200 500 1000 2000 --n-sims 100 --n-jobs "$N_JOBS"
done
python -u run_track_a.py --full-model-recovery --n-datasets 50 --n-jobs "$N_JOBS"
