
import os
import numpy as np
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from termcolor import cprint
from tqdm import tqdm
import optuna
from hydra.core.hydra_config import HydraConfig
from hydra import initialize, initialize_config_module, initialize_config_dir, compose

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

print("Script started")  # デバッグ用プリント文

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    try:
        set_seed(args.seed)
        logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir


        if args.use_wandb:
            wandb.init(mode="online", dir=logdir, project="MEG-classification")
        else:
            os.environ["WANDB_MODE"] = "dryrun"

        data_dir = args.data_dir

        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        g = torch.Generator()
        g.manual_seed(args.seed)

        loader_args = {"num_workers": args.num_workers, "worker_init_fn": seed_worker, "generator": g}

        train_set = ThingsMEGDataset("train", data_dir, front_cut=args.front_cut, tail_cut=args.tail_cut, highcut=args.highcut, pretrain = False)
        val_set = ThingsMEGDataset("val", data_dir, front_cut=args.front_cut, tail_cut=args.tail_cut, highcut=args.highcut, pretrain = False)
        test_set = ThingsMEGDataset("test", data_dir, front_cut=args.front_cut, tail_cut=args.tail_cut, highcut=args.highcut, pretrain = False)

        train_loader = CustomDataLoader(train_set, num_subjects=4, batch_size=args.batch_size, shuffle=True, pretrain = False, **loader_args)
        val_loader = CustomDataLoader(val_set, num_subjects=4, batch_size=args.batch_size, shuffle=False, pretrain = False, **loader_args)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_args)

        max_val_acc = 0
        accuracy = Accuracy(task="multiclass", num_classes=train_set.num_classes, top_k=10).to(args.device)


        # ------------------
        #   Model
        # ------------------

        model = ConvTransformerClassifier(train_set.num_classes,train_set.seq_len,train_set.num_channels,out_channels=args.transformer_out_channels,transformer_hidden_dim=args.transformer_hidden_dim,num_heads=args.num_heads,num_layers=args.num_layers,).to(args.device)
        
        # ------------------
        #   Optimizer
        # ------------------

        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=args.lr)

        # ------------------
        #   Start training
        # ------------------


        scaler = GradScaler()
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            train_loss, train_acc, val_loss, val_acc = [], [], [], []

            model.train()
            for X, y, subject_id in tqdm(train_loader, desc="Train"):
                X, y, subject_id = X.to(args.device), y.to(args.device), subject_id.to(args.device)
                optimizer.zero_grad()

                with autocast(dtype=torch.bfloat16):
                    y_pred = model(X, subject_id)
                    loss = F.cross_entropy(y_pred, y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss.append(loss.item())
                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()
            with torch.no_grad():
                for X, y, subject_id in tqdm(val_loader, desc="Validation"):
                    X, y, subject_id = X.to(args.device), y.to(args.device), subject_id.to(args.device)

                    with autocast(dtype=torch.bfloat16):
                        y_pred = model(X, subject_id)
                        v_loss = F.cross_entropy(y_pred, y)

                    val_loss.append(v_loss.item())
                    val_acc.append(accuracy(y_pred, y).item())

            print(f"Epoch {epoch+1}/{args.epochs} | train loss: {np.mean(train_loss):.3f} | train acc: {np.mean(train_acc):.3f} | val loss: {np.mean(val_loss):.3f} | val acc: {np.mean(val_acc):.3f}")
            torch.save(model.state_dict(), os.path.join(logdir, "model_last.pt"))
            if args.use_wandb:
                wandb.log({"train_loss": np.mean(train_loss), "train_acc": np.mean(train_acc), "val_loss": np.mean(val_loss), "val_acc": np.mean(val_acc)})

            if np.mean(val_acc) > max_val_acc:
                cprint("New best.", "cyan")
                torch.save(model.state_dict(), os.path.join(logdir, "model_best.pt"))
                max_val_acc = np.mean(val_acc)



        # ----------------------------------
        #  Start evaluation with best model
        # ----------------------------------

        print("Evaluating with the best model...")

        model.load_state_dict(torch.load(os.path.join(logdir, "model_best.pt"), map_location=args.device))

        preds = []
        model.eval()
        for X, subject_id in tqdm(test_loader, desc="Validation"):
            print(f"Processing batch: {X.shape}")
            print(f"Subject ID: {subject_id}")
            preds.append(model(X.to(args.device), subject_id.to(args.device)).detach().cpu())
        if len(preds) > 0:
            preds = torch.cat(preds, dim=0).numpy()
            print(f"Predictions shape: {preds.shape}")
        else:
            print("No predictions made.")

        submission_path = os.path.join(logdir, "submission.npy")
        print(f"Saving submission to {submission_path}")

        # ファイルの保存
        np.save(submission_path, preds)

        # ファイルの存在確認
        if os.path.exists(submission_path):
            cprint(f"Submission {preds.shape} saved at {submission_path}", "cyan")
        else:
            cprint(f"Failed to save submission at {submission_path}", "red")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    import sys
    from omegaconf import OmegaConf
    if 'google.colab' in sys.modules:
        from google.colab import drive
        drive.mount('/content/drive')
        # Hydraの引数競合を避けるための設定
        sys.argv = sys.argv[:1]
    run()
