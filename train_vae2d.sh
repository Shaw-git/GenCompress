#!/bin/bash
#SBATCH --job-name=model_dim4  # Job name
#SBATCH --output=logs/vae_model_dim4.out  # Std output log
#SBATCH --error=logs/vae_model_dim4.err   # Std error log
#SBATCH --mail-type=ALL                     # Email notifications for all job states
#SBATCH --mail-user=lixiao37mail@gmail.com  # Email address for notifications
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=4                   # Number of CPU cores per task
#SBATCH --mem=100000mb                       # Memory per node
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

vae_path="./snapshots/vae/laten_dim/model_dim4"
train_set="E3SM"
test_set="E3SM_test"

python train_vae2d.py \
    --save_path=$vae_path \
    --batch_size=32 \
    --iterations=400 \
    --model_dim=4 \
    --lr=0.0005 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00002\
    --sr_dim=-1\
    # --pretrain="./snapshots/vae/e3sm/train_on_5Sets/model_bs32_ep200k.pt"

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"
