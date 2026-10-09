#!/bin/bash
#SBATCH --job-name=SUMS_Imputation_10k
#SBATCH --time=60
#SBATCH --account=ivsusers
#SBATCH --output=/home/r0937090/SUMS_imput/logs/stdout.%A.%a.log
#SBATCH --error=/home/r0937090/SUMS_imput/logs/stderr.%A.%a.log
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=6G
#SBATCH --partition=ivs-short
#SBATCH --qos=normal
#SBATCH --array=1-10
module purge
module load anaconda
source activate SUMS_env
total=8250
workers=10
offset=8000
n=$((total-offset))
chunk=$(( (n + workers - 1) / workers ))
echo "SLURM ID: ${SLURM_ARRAY_TASK_ID}"
START=$(( (${SLURM_ARRAY_TASK_ID} - 1) * chunk + offset ))
END=$(( START + chunk ))
if [ "$END" -gt "$total" ]; then
    END=$total
fi
srun python SUMS_GPimputation.py "$START" "$END"
