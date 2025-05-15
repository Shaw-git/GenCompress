import os
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, ConcatDataset
from tqdm import tqdm
import argparse
import shutil
import time
from models.CDC import compress_modules_2d as compress_modules
import torch
from utils import *


def save_json(json_pth, data):
    """Save data to a JSON file, updating if it already exists."""
    if os.path.exists(json_pth):
        with open(json_pth, 'r') as json_file:
            existing_data = json.load(json_file)
        existing_data.update(data)
        data = existing_data
        
    with open(json_pth, 'w') as json_file:
        json.dump(data, json_file, indent=4)

def relative_rmse_error_ornl(x, y, axis=(1, 2, 3, 4)):
    """Compute the relative RMSE between two arrays or tensors, returns a list of floats per batch."""
    try:
        assert x.shape == y.shape
    except:
        print("x.shape != y.shape", x.shape, y.shape)
    
    if isinstance(x, torch.Tensor):
        mse  = torch.mean((x - y) ** 2, dim=axis, keepdim=True)
        maxv = torch.amax(x, dim=axis, keepdim=True)
        minv = torch.amin(x, dim=axis, keepdim=True)
        error = torch.sqrt(mse) / (maxv - minv + 1e-8)
        return error.view(-1).cpu().tolist()  # flatten and convert to list of floats

    elif isinstance(x, np.ndarray):
        mse  = np.mean((x - y) ** 2, axis=axis, keepdims=True)
        maxv = np.amax(x, axis=axis, keepdims=True)
        minv = np.amin(x, axis=axis, keepdims=True)
        error = np.sqrt(mse) / (maxv - minv + 1e-8)
        return error.reshape(-1).tolist()  # flatten and convert to list

    else:
        raise TypeError(f"Unsupported type: {type(x)}")


        

def depadding(data, max_size):
    shape = data.shape
    if shape[-1]>max_size:
        pad_left = (shape[-1]-max_size)//2
        pad_right = -1*(shape[-1]-max_size - pad_left)
        return data[...,pad_left:pad_right, pad_left:pad_right]
    else:
        return data

def train_epoch_vae(model, loader, optimizer, scheduler, criterion, loss_beta, device, iteration = 0):
    """Train the model for one epoch."""
    model.train()
    running_loss1 = 0.0
    running_loss2 = 0.0
    
    for data_dict in loader:
        
        inputs  = data_dict["input"].to(device)
        targets = inputs
 
        optimizer.zero_grad()
        results = model(inputs)
        
        outputs = results["output"]
        
        loss_mse = criterion(outputs, targets)
        loss_bpp = results["bpp"].mean()
        
        loss_bpp = loss_bpp * loss_beta
        
        loss =  (loss_mse  + loss_bpp) if loss_beta>0.0 else loss_mse
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        running_loss1 += loss_mse.item() * inputs.size(0)
        running_loss2 += loss_bpp.item() * inputs.size(0)
        
        iteration+=1
    
    epoch_loss1 = running_loss1 / len(loader.dataset)
    epoch_loss2 = running_loss2 / len(loader.dataset)
    
    return epoch_loss1, epoch_loss2, iteration

def test_epoch_vae(model, loader, criterion, device):
    """Test the model and compute the reconstruction results."""
    model.eval()
    all_data = []
    
    recons_data = torch.zeros_like(loader.dataset.data_input)

    
    bit_count = 0
    
    with torch.no_grad():
        for data_dict in loader:
            inputs  = data_dict["input"].to(device)
            targets = data_dict["input"].cpu()

            results = model(inputs)
            outputs = results["output"].detach().cpu()*data_dict["scale"] + data_dict["offset"]
            bit_count += torch.sum(results["frame_bit"].detach().cpu()).item()

            for i in range(len(inputs)):
                idx0, idx1, start_t, end_t = data_dict["index"]
                recons_data[idx0[i], idx1[i], start_t[i]:end_t[i]] = outputs[i]

    return recons_data, bit_count

class Info:
    def __init__(self,data_name, bpp = 32, model_path = None, json_path =None):
        self.json_path = json_path
        self.model_path = model_path
        
        self.data_name = data_name
        self.bpp = bpp
        self.best_nrmse = 1e10
        self.best_nrmse_cr = 0
        self.best_epoch = -1
        self.all_eval_nrmse = []
        self.all_eval_bpp = []
        self.all_eval_cr = []
        
    def save_json(self):
        save_json(self.json_path, {
            self.data_name: {
                "NRMSE": self.all_eval_nrmse,
                "best_nrmse": self.best_nrmse,
                "best_nrmse_cr": self.best_nrmse_cr,
                "best_index": self.best_epoch,
                "bpp": self.all_eval_bpp,
                "cr": self.all_eval_cr
            }
        })
        
    def save_last_model(self, model):
        torch.save(model.state_dict(), self.model_path.replace(".pt", "_final.pt"))
                    
        
    def update(self, model, epoch, nrmse, bpp, dname):
        assert(self.data_name == dname)
        self.all_eval_nrmse.append(nrmse)
        self.all_eval_bpp.append(bpp)
        self.all_eval_cr.append(self.bpp/bpp)
        
        if nrmse <= self.best_nrmse:
            torch.save(model.state_dict(), self.model_path)
            self.best_nrmse = nrmse
            self.best_nrmse_cr = self.bpp/bpp
            self.best_epoch = epoch
            
        self.save_json()
        
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
    parser.add_argument('--end_beta', type=float, default=2e-5, help='loss beta')
    
    parser.add_argument('--beta_start', type=float, default=0.75, help='loss beta')
    parser.add_argument('--model_dim', type=int, default=16, help='loss beta')
    parser.add_argument('--pretrain', type=str, default="", help='pretrain path')
    
    # Datatset
    parser.add_argument('--train_set', type=str, default="S3D")
    parser.add_argument('--test_set', type=str, default="E3SM_test")
    parser.add_argument('--config', type=str, default="./configs/config_vae.yaml")

    args = parser.parse_args()
    
        
    return args

        

if __name__ == "__main__":
    args = get_argument()
    
    save_path = args.save_path

# Ensure save path exists
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    shutil.copy(args.config, save_path+"/config_vae.yaml")

    # Paths for model and JSON files
    model_path = os.path.join(save_path, f"model_bs{args.batch_size}_ep{args.iterations}k.pt")
    json_path  = os.path.join(save_path, f"model_bs{args.batch_size}_ep{args.iterations}k.json")
    
    args.iterations = args.iterations * 1000
    save_json(json_path, {"argument":vars(args)})

    train_args = convert_args(args, train=True)
    
    print(train_args)
    
    
    train_datasets = build_dataset(train_args, syn_length = True)
    print("Length for Each dataset", [len(dataset) for dataset in train_datasets])
    merged_dataset = ConcatDataset(train_datasets)
    
    train_loader = DataLoader(merged_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    
    

    test_args = convert_args(args, train=False)
    test_datasets = build_dataset(test_args, syn_length = False)
    test_loaders = [DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=3) for dataset in test_datasets]
    # Model and device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # main(args)
    
    model = compress_modules.ResnetCompressor(  dim= args.model_dim,
                                                dim_mults=[1,2,3,4],
                                                reverse_dim_mults=[4,3,2,1],
                                                hyper_dims_mults=[4,4,4],
                                                channels = 1,
                                                out_channels = 1,
                                                d3=False)
    
    if args.pretrain != "":
        print("Load pretrain model:", args.pretrain)
        state_dict = torch.load(args.pretrain)
        model.load_state_dict(state_dict)
        
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = nn.DataParallel(model)  # Wrap model with DataParallel for multi-GPU
    else:
        print("Using a single GPU!")
        
    
    model = model.to(device)
    # Loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(i/5*args.iterations) for i in range(1,5)], gamma=args.lr_gamma)
    
    test_names = [loader.dataset.dataset_name for loader in test_loaders]
    loggers = {name:Info(name, 32, model_path, json_path) for name in  test_names}
    
    cur_iters = 0
    is_eval = np.zeros(100, dtype=bool)
    
    print(f"Learning rate milestones: {[int(i/5*args.iterations) for i in range(1, 5)]}")
    
#     estimate the remaining time
    start_time = time.time()
    
    
    while cur_iters < args.iterations:
       

        beta = args.init_beta if cur_iters < (args.iterations*args.beta_start) else args.end_beta
        
        
        
        mse_loss, bbp_loss, cur_iters = train_epoch_vae(model, train_loader, optimizer, scheduler, criterion, beta , device, cur_iters)
        train_loss = mse_loss + bbp_loss
        
        
    
        eval_index = cur_iters // (args.iterations//100)
        if not is_eval[eval_index]:
            is_eval[eval_index] = True
          
        
            for test_loader in test_loaders:
                cur_dataset = test_loader.dataset
                dname = cur_dataset.dataset_name
                original_data = cur_dataset.original_data()
                
                recons_data, bit_count = test_epoch_vae(model, test_loader, criterion, device)
                recons_data = cur_dataset.deblocking_hw(recons_data)
                
                bpp = float(bit_count/np.prod(recons_data.shape))
                
                nrmse = relative_rmse_error_ornl(original_data, recons_data)
                nrmse = float(nrmse[0])

                loggers[dname].update(model, cur_iters, nrmse, bpp, dname)
                
                loggers[dname].save_last_model(model)
                
                print(dname, f"Progress: {eval_index}/100 ,  Iter {cur_iters}, Train Loss: {train_loss:.6f} ({mse_loss:.6f} + {bbp_loss:.6f})", 
                             "NRMSE:",nrmse, f"BPP: {bpp:.6f} CR: {32/bpp:.6f}")
                
                total_time = time.time() - start_time
                remaining_time = (args.iterations - cur_iters) * (total_time/cur_iters)
                print(f"Training time: {'%d:%d:%d'%(second_to_time(remaining_time))}/{'%d:%d:%d'%(second_to_time(total_time))}")
        
            print()
            
    print("Training complete.")

