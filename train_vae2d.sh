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
module load pytorch/1.8.1

T1=$(date +%s)
# --train_set="S3D,JHTDB,Hurricane,ERA5,Sunquake,Blastnet" \
# Run the VAE3D training script

vae_path="./snapshots/vae/e3sm/train_on_SHPJ"
train_set="Hurricane,S3D,JHTDB,PDE"
test_set="E3SM_test"

python train_vae2d.py \
    --save_path=$vae_path \
    --batch_size=32 \
    --iterations=400 \
    --model_dim=16 \
    --lr=0.0005 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00002

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"
