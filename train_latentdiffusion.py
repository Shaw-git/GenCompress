# from models.video_diffusion_rcond import GaussianDiffusion, Unet3D, Trainer
from models.video_diffusion_interpo import GaussianDiffusion, Unet3D, Trainer
from mydataset.e3sm_latent import ClimateLatent
from tools_online.io.json_io import save_json

time_steps = 1000

interpo_rate = 0
results_folder = "./results/E3SM_prediction/VD_Latent_interpo_%d_t%d_first4_last1"%(interpo_rate,time_steps)
dataset_arg = {"data_path":"./data/e3sm100", "n_frame":16, "inst_norm": True, "norm_type":'range2'}

dataset = ClimateLatent(dataset_arg)

import torch

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


import torch

model = Unet3D(
    dim = 64,
    out_dim = 64,
    channels = 64,
    dim_mults = (1, 2, 4, 8),
    use_bert_text_cond=False
)

diffusion = GaussianDiffusion(
    model,
    image_size = 16,
    num_frames = 10,
    timesteps = time_steps,   # number of steps
    loss_type = 'l2'    # L1 or L2
).cuda()


# load_matching_parameters(diffusion, "./results/VD_LatentRandCond_InstRange2Norm_10/model-450.pt")
# load_matching_parameters(diffusion, "./results/VD_Latent_InstRange2Norm_16_e3sm100_pretrain/model-499.pt")


trainer = Trainer(
    diffusion,
    dataset,                         # this folder path needs to contain all your training data, as .gif files, of correct image size and number of frames
    train_batch_size = 64,
    train_lr = 1e-4,
    save_and_sample_every = 10000,
    train_num_steps = 300000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    results_folder = results_folder,
    interpo_rate = interpo_rate,
    cond_idx = torch.LongTensor([0,1,2,3,15])
)

save_json(results_folder + "/train.json", {"dataset":dataset_arg})

trainer.train()