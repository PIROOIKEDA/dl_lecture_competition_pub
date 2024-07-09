#main
#前処理の引数を追加
#モデルをLSTMに変更
#optuna追加
# Objective関数内でのエポック数設定とHydraの初期化


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

from src.datasets import EnhancedEEGDataset
from src.datasets import ThingsMEGDataset
from src.datasets import CustomDataLoader
from src.models import SimpleConvLSTMClassifier
from src.models import ConvLSTMClassifier
from src.models import SimpleConvLSTMEncoder2
from src.models import ParallelConvLSTMClassifier
from src.utils import set_seed
import yaml
from hydra.core.global_hydra import GlobalHydra
from torch.utils.data import Dataset, DataLoader, Subset, Sampler
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim



class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path=None):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            if path:
                self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            if path:
                self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss

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
        loader_args = {"num_workers": args.num_workers}

        train_set = ThingsMEGDataset("train", data_dir)
        print(f"Train set size: {len(train_set)}")

        val_set = ThingsMEGDataset("val", data_dir)
        print(f"Validation set size: {len(val_set)}")

        test_set = ThingsMEGDataset("test", data_dir)
        print(f"Test set size: {len(test_set)}")


        sample_data, sample_label, sample_subject_id = train_set[0]

        print(f"Sample data shape: {sample_data.shape}")  # (271, 281)
        print(f"Sample label shape: {sample_label.shape}")  # () - integer value
        print(f"Sample subject ID shape: {sample_subject_id.shape}")  # () - integer value

       
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)
        
        g = torch.Generator()
        g.manual_seed(args.seed) # num_worker>1より再現性確保の設定必要

        train_loader = CustomDataLoader(train_set, num_subjects=4, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        val_loader = CustomDataLoader(val_set, num_subjects=4, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, worker_init_fn=seed_worker, generator=g)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **loader_args,  worker_init_fn=seed_worker, generator=g)

        # デバッグ用プリント
        for X_batch, y_batch, subject_id_batch in train_loader:
            print(f"X_batch shape: {X_batch.shape}")  # Expected: (32, 271, 281)
            print(f"y_batch shape: {y_batch.shape}")  # Expected: (32,)
            print(f"subject_id_batch shape: {subject_id_batch.shape}")  # Expected: (32,)
            break


        # ------------------
        #   Grid Search
        # ------------------
        max_val_acc = 0
        accuracy = Accuracy(
            task="multiclass", num_classes=train_set.num_classes, top_k=10
        ).to(args.device)

        early_stopping = EarlyStopping(patience=20, verbose=True)
        early_stopping_grid = EarlyStopping(patience=3, verbose=True)   

        best_params ={'out_channels':64,  'lstm_hidden_dim':64}
        print(f'Best Params: {best_params}')

        # ------------------
        #       Model
        # ------------------

        print("Initializing model...")
        
        model = ConvLSTMClassifier(train_set.num_classes, train_set.seq_len-35, train_set.num_channels).to(args.device)

        # ------------------
        # Optimizer
        # ------------------
        print("Setting up optimizer...")
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

        # ------------------
        #   Start training
        # ------------------


        for epoch in range(args.epochs):
            print(f"Epoch {epoch+1}/{args.epochs}")

            train_loss, train_acc, val_loss, val_acc = [], [], [], []

            model.train()

            for X, y, subject_id in tqdm(train_loader, desc="Train"):

                X, y, subject_id = X.to(args.device), y.to(args.device), subject_id.to(args.device) #subject_idを追加
          
                y_pred = model(X, subject_id)
                loss = F.cross_entropy(y_pred, y)

                train_loss.append(loss.item())

                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                acc = accuracy(y_pred, y)
                train_acc.append(acc.item())

            model.eval()

            for X, y, subject_id in tqdm(val_loader, desc="Validation"):
                X, y, subject_id = X.to(args.device), y.to(args.device), subject_id.to(args.device)


                with torch.no_grad():
                    y_pred = model(X, subject_id)

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

            early_stopping(np.mean(val_loss), model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

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
