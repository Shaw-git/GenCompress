#!/bin/bash
#SBATCH --job-name=SHPJ  # Job name
#SBATCH --output=logs/SHPJ.out  # Std output log
#SBATCH --error=logs/SHPJ.err   # Std error log
#SBATCH --mail-type=ALL                     # Email notifications for all job states
#SBATCH --mail-user=lixiao37mail@gmail.com  # Email address for notifications
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=80000mb                       # Memory per node
#SBATCH --partition=gpu                     # GPU partition
#SBATCH --gpus=a100:1                       # Number of GPUs (A100)
#SBATCH --time=100:00:00                     # Maximum job runtime

echo "Date       = $(date)"
echo "Host       = $(hostname -s)"
echo "Directory  = $(pwd)"

module purge
module load pytorch/2.2.0

T1=$(date +%s)
# --train_set="S3D,JHTDB,Hurricane,ERA5,Sunquake,Blastnet" \
# Run the VAE3D training script
python train_latentdiffusion.py \
    --iterations=300 \
    --time_steps=1000 \
    --interpo_rate=3 \
    --train_set="PDE" \
    --test_set="E3SM_test" \
    --keyframe_pretrain="./snapshots/vae/e3sm/train_on_SHPJ/model_bs32_ep400k.pt" \
    --result_path="./snapshots/latent_model/e3sm/SHPJ_BlockInstNorm"\

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"

