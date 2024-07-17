import os, sys
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig
import wandb
from termcolor import cprint
from tqdm import tqdm

from src.datasets import ThingsMEGDataset
from src.datasets import CustomDataLoader
from src.models import ConvTransformerClassifier
from src.utils import set_seed
import yaml
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import random
from torch.cuda.amp import GradScaler, autocast
import bitsandbytes as bnb



@torch.no_grad()
@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    set_seed(args.seed)
    savedir = os.path.dirname(args.model_path)

    data_dir = args.data_dir
    
    # ------------------
    #    Dataloader
    # ------------------ 
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)   
    g = torch.Generator()
    g.manual_seed(args.seed)


    loader_args = {"num_workers": args.num_workers, "worker_init_fn": seed_worker, "generator": g}

    test_set = ThingsMEGDataset("test", data_dir, front_cut=args.front_cut, tail_cut=args.tail_cut, highcut=args.highcut, pretrain = False)

    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_args)

    # ------------------
    #       Model
    # ------------------
    model = ConvTransformerClassifier(test_set.num_classes,test_set.seq_len,test_set.num_channels,out_channels=args.transformer_out_channels,transformer_hidden_dim=args.transformer_hidden_dim,num_heads=args.num_heads,num_layers=args.num_layers,).to(args.device)
    model.load_state_dict(torch.load(os.path.join(savedir, "model_best.pt"), map_location=args.device))


    # ------------------
    #  Start evaluation
    # ------------------ 
    preds = [] 
    model.eval()
    for X, subject_idxs in tqdm(test_loader, desc="Validation"):        
        preds.append(model(X.to(args.device), subject_idxs.to(args.device)).detach().cpu())
        
    preds = torch.cat(preds, dim=0).numpy()
    np.save(os.path.join(savedir, "submission"), preds)
    cprint(f"Submission {preds.shape} saved at {savedir}", "cyan")


if __name__ == "__main__":
    run()