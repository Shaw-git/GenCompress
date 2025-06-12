#!/bin/bash
#SBATCH --job-name=JSE_VAE3D_SR_from_scratch  # Job name
#SBATCH --output=logs/JSE_VAE3D_SR_from_scratch.out  # Std output log
#SBATCH --error=logs/JSE_VAE3D_SR_from_scratch.err   # Std error log
#SBATCH --mail-type=ALL                     # Email notifications for all job states
#SBATCH --mail-user=lixiao37mail@gmail.com  # Email address for notifications
#SBATCH --nodes=1                           # Number of nodes
#SBATCH --ntasks=1                          # Number of tasks (processes)
#SBATCH --cpus-per-task=10                   # Number of CPU cores per task
#SBATCH --mem=100000mb                       # Memory per node
#SBATCH --partition=gpu                     # GPU partition
#SBATCH --gpus=a100:2                       # Number of GPUs (A100)
#SBATCH --time=100:00:00                     # Maximum job runtime

echo "Date       = $(date)"
echo "Host       = $(hostname -s)"
echo "Directory  = $(pwd)"

module purge
module load pytorch/1.8.1

T1=$(date +%s)

# vae_path="./snapshots/vae3d/e3sm/VAE3D_on_5Sets_NSR"
# train_set="Hurricane,PDE,ERA5,TUM,HYCOM"
# test_set="E3SM_test"

# python train_vae3d.py \
#     --save_path=$vae_path \
#     --batch_size=32 \
#     --iterations=100 \
#     --model_dim=16 \
#     --lr=0.0005 \
#     --beta_start=0.5 \
#     --train_set=$train_set \
#     --test_set=$test_set \
#     --init_beta=0.00001 \
#     --end_beta=0.00002\
#     --sr_dim=-1\
#     --pretrain="./snapshots/vae3d/e3sm/SHPJ_VAE3D_NSR_2GPU/model_bs32_ep400k.pt"
    
    
    
vae_path="./snapshots/vae3d/JSE_ds/VAE3D_SR_from_scratch"
train_set="JHTDB"
test_set="JHTDB"

python train_vae3d.py \
    --save_path=$vae_path \
    --batch_size=32 \
    --iterations=300 \
    --model_dim=16 \
    --lr=0.0004 \
    --beta_start=0.5 \
    --train_set=$train_set \
    --test_set=$test_set \
    --init_beta=0.00001 \
    --end_beta=0.00002\
    --sr_dim=16\
    # --pretrain="./snapshots/vae3d/e3sm/VAE3D_on_5Sets_SR/model_bs32_ep100k_final.pt"
    

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = $ELAPSED seconds"
