#dataset

import os
import torch
import numpy as np
from scipy.signal import resample, butter, filtfilt
from torch.utils.data import Dataset

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", new_fs=None, lowcut=None, highcut=None, baseline_correct=False) -> None:
        """
        Initialize dataset and load data.
        """
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # Load the data files
        try:
            print(f"Loading {split}_X.pt")
            file_path = os.path.join(data_dir, f"{split}_X.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            self.X = torch.load(file_path)

            print(f"Loading {split}_subject_idxs.pt")
            file_path = os.path.join(data_dir, f"{split}_subject_idxs.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            self.subject_idxs = torch.load(file_path)

            if split in ["train", "val"]:
                print(f"Loading {split}_y.pt")
                file_path = os.path.join(data_dir, f"{split}_y.pt")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_path} does not exist.")
                self.y = torch.load(file_path)
                assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise

        print(f"Loaded dataset: {split} with {len(self.X)} samples.")

    def preprocess(self, new_fs=None, lowcut=None, highcut=None, baseline_correct=False, old_fs=1000):
        """
        Preprocess the data.
        """
        X_np = self.X.numpy()

        if baseline_correct:
            baseline = np.mean(X_np, axis=-1, keepdims=True)
            X_np = X_np - baseline

        if new_fs is not None and new_fs != old_fs:
            num_samples = int(X_np.shape[-1] * (new_fs / old_fs))
            X_np = resample(X_np, num_samples, axis=-1)

        if lowcut is not None:
            b, a = butter(5, lowcut / (0.5 * new_fs), btype='high')
            X_np = filtfilt(b, a, X_np, axis=-1)

        if highcut is not None:
            b, a = butter(5, highcut / (0.5 * new_fs), btype='low')
            X_np = filtfilt(b, a, X_np, axis=-1)

        self.X = torch.tensor(X_np)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i):
        if hasattr(self, "y"):
            return self.X[i], self.y[i], self.subject_idxs[i]
        else:
            return self.X[i], self.subject_idxs[i]

    @property
    def num_channels(self) -> int:
        return self.X.shape[1]

    @property
    def seq_len(self) -> int:
        return self.X.shape[2]

