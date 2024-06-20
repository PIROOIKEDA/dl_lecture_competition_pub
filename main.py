import os
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
from src.models import BasicConvClassifier
from src.utils import set_seed

print("Script started")  # デバッグ用プリント文

@hydra.main(version_base=None, config_path="configs", config_name="config")
def run(args: DictConfig):
    try:
        print("Inside run function")  # デバッグ用プリント文
        set_seed(args.seed)
        logdir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
        print(f"Output directory: {logdir}")
        
        if args.use_wandb:
            wandb.init(mode="online", dir=logdir, project="MEG-classification")
        else:
            os.environ["WANDB_MODE"] = "dryrun"

        # データディレクトリを設定
        data_dir = args.data_dir
        print(f"Data directory: {data_dir}")

        # ------------------
        #    Dataloader
        # ------------------
        print("Loading datasets...")
        loader_args = {"batch_size": args.batch_size, "num_workers": args.num_workers}
        
        train_set = ThingsMEGDataset("train", data_dir)
        print(f"Train set size: {len(train_set)}")
        train_loader = torch.utils.data.DataLoader(train_set, shuffle=True, **loader_args)
        
        val_set = ThingsMEGDataset("val", data_dir)
        print(f"Validation set size: {len(val_set)}")
        val_loader = torch.utils.data.DataLoader(val_set, shuffle=False, **loader_args)
        
        test_set = ThingsMEGDataset("test", data_dir)
        print(f"Test set size: {len(test_set)}")
        test_loader = torch.utils.data.DataLoader(test_set, shuffle=False, batch_size=args.batch_size, num_workers=args.num_workers)

        # ------------------
        #       Model
        # ------------------
        print("Initializing model...")
        model = BasicConvClassifier(
            train_set.num_classes, train_set.seq_len, train_set.num_channels
        ).to(args.device)

        # ------------------
        #     Optimizer
        # ------------------
        print("Setting up optimizer...")
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        # ------------------
        #   Start training
        # ------------------  
        max_val_acc = 0
        accuracy = Accuracy(
            task="multiclass", num_classes=train_set.num_classes, top_k=10
        ).to(args.device)
        
        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")
            
            train_loss, train_acc, val_loss, val_acc = [], [], [], []
            
            model.train()
            for X, y, subject_idxs in tqdm(train_loader, desc="Train"):
                X, y = X.to(args.device), y.to(args.device)

                y_pred = model(X)
                
                loss = F.cross_entropy(y_pred, y)
                train_loss.append(loss.item())
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()
            for X, y, subject_idxs in tqdm(val_loader, desc="Validation"):
                X, y = X.to(args.device), y.to(args.device)
                
                with torch.no_grad():
                    y_pred = model(X)
                
                val_loss.append(F.cross_entropy(y_pred, y).item())
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
        for X, subject_idxs in tqdm(test_loader, desc="Validation"):
            print(f"Processing batch: {X.shape}")
            preds.append(model(X.to(args.device)).detach().cpu())
        
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
