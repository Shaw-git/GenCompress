# from models.video_diffusion_rcond import GaussianDiffusion, Unet3D, Trainer
from models.latent_diffusion import GaussianDiffusion, Unet3D, Trainer
from models.CDC import keyframe_compressor as compress_modules
from torch.utils.data import DataLoader, ConcatDataset
import argparse
from tools_online.io.json_io import save_json
import torch
from utils import *

def load_matching_parameters(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path)["ema"]
    model_dict = model.state_dict()
    pretrained_dict = {
        k: v for k, v in checkpoint.items() 
        if k in model_dict and model_dict[k].shape == v.shape
    }

    # Update model's state dict
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict, strict=False)


def load_pretrain_keyframe_model(args):
    keyframe_model = compress_modules.ResnetCompressor(dim = 16,
                                                dim_mults=[1,2,3,4],
                                                reverse_dim_mults=[4,3,2,1],
                                                hyper_dims_mults=[4,4,4],
                                                channels = 1,
                                                out_channels = 1)
    
    print("Load pretrain model:", args.keyframe_pretrain)
    state_dict = torch.load(args.keyframe_pretrain)
    keyframe_model.load_state_dict(state_dict)
    return keyframe_model
    
def get_args():
    """
    Parses command-line arguments and returns them.
    """
    parser = argparse.ArgumentParser(description="Training script with arguments")
    
    parser.add_argument("--time_steps", type=int, default=1000, help="Number of time steps")
    parser.add_argument("--interpo_rate", type=int, default=3, help="Interpolation rate")
    parser.add_argument("--iterations", type=int, default=300, help="iterations")
    
    parser.add_argument("--pretrain", type=str, 
                        default="/home/xiao.li/DiffusionModel/results/S3D/VD_Latent_Original_InstNorm_16_S3D_pretrain_t1000_interpo_3/model-30.pt", 
                        help="Path to the pre-trained model")
    
    parser.add_argument("--keyframe_pretrain", type=str, 
                        default="/home/xiao.li/DiffusionModel/snapshots/MaskedAE/NoMask/model_bs16_ep500k.pt",
                        help="Path to the pre-trained keyframe model")
    
    parser.add_argument("--result_path", type=str, 
                        default="./snapshots/latent_model/e3sm/SHPJ_BlockInstNorm",
                        help="Path to the pre-trained keyframe model")
    
    # Datatset
    parser.add_argument('--train_set', type=str, default="E3SM")
    parser.add_argument('--test_set', type=str, default="E3SM")
    parser.add_argument('--config', type=str, default="./configs/config_latent.yaml")
    # parser.add_argument('--syn_length', type=int, default=0)
    
    args = parser.parse_args()

    args.result_path = args.result_path+"%d_Steps_%d_Interpo"%(args.time_steps, args.interpo_rate)   
    
    return args

    
if __name__ == "__main__":
    
    args = get_args()
    results_folder = args.result_path
    
    keyframe_model = load_pretrain_keyframe_model(args).cuda()
    
    
    latent_model = Unet3D(
        dim = 64,
        out_dim = 64,
        channels = 64,
        dim_mults = (1, 2, 4, 8),
        use_bert_text_cond=False
    )

    diffusion = GaussianDiffusion(
        latent_model,
        image_size = 16,
        num_frames = 10,
        timesteps = args.time_steps,   # number of steps
        loss_type = 'l2'    # L1 or L2
    ).cuda()

    # load_matching_parameters(diffusion, "./results/VD_LatentRandCond_InstRange2Norm_10/model-450.pt")

    load_matching_parameters(diffusion, args.pretrain)
    
    
    train_args = convert_args(args, train=True)
    train_datasets = build_dataset(train_args, syn_length = True)
    
    print("Length for Each Dataset", [len(dataset) for dataset in train_datasets])
    dataset = ConcatDataset(train_datasets)
    

    trainer = Trainer(
        keyframe_model,
        diffusion,
        dataset,                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
        train_batch_size = 64,
        train_lr = 1e-4,
        save_and_sample_every = 10000,
        train_num_steps = 1000 * args.iterations,         # total training steps
        gradient_accumulate_every = 1,    # gradient accumulation steps
        ema_decay = 0.995,                # exponential moving average decay
        amp = True,                        # turn on mixed precision
        results_folder = results_folder,
        interpo_rate = args.interpo_rate,
    )

    save_json(results_folder + "/train.json", {"argument":vars(args)})

    trainer.train()