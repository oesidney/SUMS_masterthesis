#!/bin/bash
#SBATCH --job-name=SUMS_Imputation_10k
#SBATCH --time=1200
#SBATCH --account=ivsusers
#SBATCH --output=/home/r0937090/SUMS_imput/logs/stdout.%A.%a.log
#SBATCH --error=/home/r0937090/SUMS_imput/logs/stderr.%A.%a.log
#SBATCH --cpus-per-task=4
#SBATCH --ntasks=1
#SBATCH --mem=32G
#SBATCH --partition=ivs-long
#SBATCH --qos=normal
#SBATCH --array=1-40
module purge
module load anaconda
source activate SUMS_env
total=10000
workers=40
offset=0
n=$((total-offset))
chunk=$(( (n + workers - 1) / workers ))
echo "SLURM ID: ${SLURM_ARRAY_TASK_ID}"
START=$(( (${SLURM_ARRAY_TASK_ID} - 1) * chunk + offset ))
END=$(( START + chunk ))
if [ "$END" -gt "$total" ]; then
    END=$total
fi
srun python SUMS_GPimputation.py "$START" "$END"
