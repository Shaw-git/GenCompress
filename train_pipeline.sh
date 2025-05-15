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
#SBATCH --time=100:00:00                    # Maximum job runtime

echo "Date       = $(date)"
echo "Host       = $(hostname -s)"
echo "Directory  = $(pwd)"

module purge
module load pytorch/2.2.0

T1=$(date +%s)

# Hyperparameters and paths
iterations=1
batch_size=32

vae_path="./snapshots/vae/e3sm/test"
vae_result_model="${vae_path}/model_bs${batch_size}_ep${iterations}k.pt"

diffusion_pretrain="/home/xiao.li/DiffusionModel/results/S3D/VD_Latent_Original_InstNorm_16_S3D_pretrain_t1000_interpo_3/model-30.pt"
diffusion_result_path="./snapshots/latent_model/e3sm/test"

train_set="S3D"
test_set="E3SM_test"

# Run VAE2D training
python train_vae2d.py \
    --save_path="${vae_path}" \
    --batch_size="${batch_size}" \
    --iterations="${iterations}" \
    --model_dim=16 \
    --lr=0.0005 \
    --beta_start=0.5 \
    --train_set="${train_set}" \
    --test_set="${test_set}" \
    --init_beta=0.00001 \
    --end_beta=0.00002

# Run latent diffusion model training
python train_latentdiffusion.py \
    --iterations=300 \
    --time_steps=1000 \
    --interpo_rate=3 \
    --train_set="${train_set}" \
    --test_set="${test_set}" \
    --keyframe_pretrain="${vae_result_model}" \
    --pretrain="${diffusion_pretrain}" \
    --result_path="${diffusion_result_path}"

T2=$(date +%s)

ELAPSED=$((T2 - T1))
echo "Elapsed Time = ${ELAPSED} seconds"
