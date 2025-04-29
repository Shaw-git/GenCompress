import numpy as np
import argparse
import copy

def get_argument():
    parser = argparse.ArgumentParser(description="Train a UNet with Channel Attention model.")
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size for training')
    parser.add_argument('--save_path', type=str, default="./snapshots/E3SM/E3SM_VAE", help='Path to save model and results')
    # parser.add_argument('--epochs', type=int, default=600, help='Number of epochs for training')
    parser.add_argument('--iterations', type=int, default=400, help='Number of epochs for training')
    parser.add_argument('--begin_sr', type=int, default=0, help='Number of epochs for training')
    parser.add_argument('--sr_type', type=str, default="BCRN", help='Number of epochs for training')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # parser.add_argument('--lr_milestones', type=int, nargs='+', default=[100, 450], help='Learning rate milestones')
    parser.add_argument('--lr_gamma', type=float, default=0.5, help='Learning rate gamma')
    
    parser.add_argument('--init_beta', type=float, default=1e-5, help='loss beta')
    parser.add_argument('--end_beta', type=float, default=1e-4, help='loss beta')
    
    parser.add_argument('--beta_start', type=float, default=0.75, help='loss beta')
    parser.add_argument('--model_dim', type=int, default=16, help='loss beta')
    parser.add_argument('--pretrain', type=str, default="", help='pretrain path')
    
    # Datatset
    parser.add_argument('--train_set', type=str, default="NYX,S3D")
    parser.add_argument('--test_set', type=str, default="E3SM,JHTDB")
    parser.add_argument('--config', type=str, default="./configs/config.yaml")
    # parser.add_argument('--syn_length', type=int, default=0)

    parser.add_argument('--input_size', type=int, default=256, help='data size')
    parser.add_argument('--n_frame', type=str, default="8", help='data size')
    

    args = parser.parse_args()
    
    args.n_frame =  np.asarray(args.n_frame.split(","), dtype = np.int32).tolist()
    if len(args.n_frame) == 1:
        args.n_frame = args.n_frame[0]
        
    return args


def build_dataset(dargs, syn_length=False):
    all_datasets = []
    for name in dargs:
        if "E3SM" in name:
            from mydataset.e3sm240 import Climate240
            all_datasets.append(Climate240(dargs[name]))
        elif "S3D" ==name:
            from mydataset.s3d_dataset import S3D
            all_datasets.append(S3D(dargs[name]))
        elif "S3D300" ==name:
            from mydataset.s3d300_dataset import S3D300
            all_datasets.append(S3D300(dargs[name]))
        elif "JHTDB" == name:
            from mydataset.jhtdb_dataset import JHTDB
            all_datasets.append(JHTDB(dargs[name]))
        elif "NYX" == name:
            from mydataset.nyx_dataset import NYX
            all_datasets.append(NYX(dargs[name]))
        elif "Hurricane" == name:
            from mydataset.hurricane_dataset import Hurricane
            all_datasets.append(Hurricane(dargs[name]))
        elif "Video" == name:
            from mydataset.video_dataset import Video
            all_datasets.append(Video(dargs[name]))
        elif "ERA5" == name:
            from mydataset.era5_dataset import ERA5
            all_datasets.append(ERA5(dargs[name]))
        elif "Sunquake" == name:
            from mydataset.sunquake_dataset import Sunquake
            all_datasets.append(Sunquake(dargs[name]))
        elif "Blastnet" == name:
            from mydataset.blastnet_dataset import Blastnet
            all_datasets.append(Blastnet(dargs[name]))
        else:
            print("NotImplementDataset:", name)
            assert NotImplementDataset
            
    if syn_length:
        max_length = np.max([len(d) for d in all_datasets])
        
        for d in all_datasets:
            d.visble_length = max_length
            
    return all_datasets


def convert_args(args, train=True):
    import yaml

# Load YAML configuration
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    if train:
        dnames = args.train_set.split(",")
        var_name = "train_var"
        norm_type = "train_norm"
        augment_type = "train_aug"
    else:
        dnames = args.test_set.split(",")
        var_name = "test_var"
        norm_type = "test_norm"
        augment_type = "test_aug"
        
    all_args = {}
    
    
    
    for dname in dnames:
        n_size  = args.input_size
        n_frame = args.n_frame
    
        cfg = config[dname]
        
        if not train:
            n_size = cfg["test_size"]
            n_frame = cfg["block_size"]
        
        train_dict = {}
        train_dict.update(cfg)
        
        train_dict.update({"var":cfg[var_name], "input_size": n_size,  "n_frame": n_frame, "norm_type":cfg[norm_type]})
        

            
        if augment_type in cfg:
            train_dict.update({"augment_type": cfg[augment_type]})
            
        all_args[dname]=train_dict
        
    return all_args


def second_to_time(remaining_time):
    remaining_hours = int(remaining_time // 3600)
    remaining_minutes = int((remaining_time % 3600) // 60)
    remaining_seconds = int(remaining_time % 60)
    return remaining_hours, remaining_minutes, remaining_seconds