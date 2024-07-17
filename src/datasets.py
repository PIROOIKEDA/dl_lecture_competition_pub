import os
import torch
import numpy as np
from scipy.signal import resample, butter, filtfilt, welch
from statsmodels.tsa.ar_model import AutoReg
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, default_collate
import random
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

import random
import torch
from torch.utils.data import DataLoader, default_collate

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, num_subjects, batch_size, shuffle=True, pretrain=True, **kwargs):
        self.dataset = dataset
        self.num_subjects = num_subjects
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.pretrain = pretrain
        self.subject_indices = {i: [] for i in range(num_subjects)}

        # Call the DataLoader __init__ method with necessary arguments
        super().__init__(dataset, batch_size=batch_size, shuffle=False, **kwargs)

        self._prepare_indices()
        print(f"Initialized CustomDataLoader with batch_size={self.batch_size} and num_subjects={self.num_subjects}")

    def _prepare_indices(self):
        for idx in range(len(self.dataset)):
            if self.pretrain:
                _, _, _, subject_id = self.dataset[idx]
            else:
                _, _, subject_id = self.dataset[idx]
            self.subject_indices[subject_id.item()].append(idx)

    def __iter__(self):
        if self.shuffle:
            for subject_id in self.subject_indices:
                random.shuffle(self.subject_indices[subject_id])

        min_samples = min(len(indices) for indices in self.subject_indices.values())
        print(f"Min samples per subject: {min_samples}")
        print(f"Batch size: {self.batch_size}, Num subjects: {self.num_subjects}")
        num_batches = min_samples // (self.batch_size // self.num_subjects)
        print(f"Number of batches: {num_batches}")

        batch_indices = []
        for i in range(num_batches):
            batch = []
            for subject_id in range(self.num_subjects):
                start = i * (self.batch_size // self.num_subjects)
                end = start + (self.batch_size // self.num_subjects)
                selected_indices = self.subject_indices[subject_id][start:end]
                batch.extend(selected_indices)
            random.shuffle(batch)
            batch_indices.append(batch)

        random.shuffle(batch_indices)
        for batch in batch_indices:
            yield default_collate([self.dataset[idx] for idx in batch])

    def __len__(self):
        min_samples = min(len(indices) for indices in self.subject_indices.values())
        return min_samples // (self.batch_size // self.num_subjects)


def preprocess_chunk(chunk, new_fs=None, lowcut=None, highcut=None, baseline_correct=False, old_fs=200):
    # Ensure a contiguous copy of the tensor before converting to numpy array
    X_np = chunk.cpu().numpy()
    print(f"X_np shape: {X_np.shape}, new_fs: {new_fs}, lowcut: {lowcut}, highcut: {highcut}, baseline_correct: {baseline_correct}, old_fs: {old_fs}")

    if baseline_correct:
        baseline = np.mean(X_np, axis=-1, keepdims=True)
        X_np = X_np - baseline

    if new_fs is not None and new_fs != old_fs:
        num_samples = int(X_np.shape[-1] * (new_fs / old_fs))
        X_np = resample(X_np, num_samples, axis=-1)

    if lowcut is not None:
        nyquist = 0.5 * new_fs if new_fs is not None else 0.5 * old_fs
        low = lowcut / nyquist
        b, a = butter(5, low, btype='high')
        X_np = filtfilt(b, a, X_np, axis=-1)

    if highcut is not None:
        nyquist = 0.5 * new_fs if new_fs is not None else 0.5 * old_fs
        high = highcut / nyquist
        b, a = butter(5, high, btype='low')
        X_np = filtfilt(b, a, X_np, axis=-1)

    X_np = X_np.copy()  # Create a copy to avoid modifying the original array
    X_tensor = torch.tensor(X_np, dtype=torch.float32)  # Convert to float32
    del X_np  # Free up memory by deleting the original numpy array
    return X_tensor

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", image_dir: str = "/content/extracted_image/Images", front_cut: int = 40, tail_cut=282, pretrain=True, new_fs=None, lowcut=None, highcut=None, baseline_correct=False, old_fs=200, chunk_size=1000) -> None:
        super().__init__()
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854
        self.front_cut = front_cut
        self.tail_cut = tail_cut
        self.image_dir = image_dir
        self.pretrain = pretrain
        self.new_fs = new_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.baseline_correct = baseline_correct
        self.old_fs = old_fs
        self.chunk_size = chunk_size
        self.data_dir = data_dir
        self.processed_X = []
        self.load_and_preprocess_data()

    def load_and_preprocess_data(self):
        try:
            print(f"Loading {self.split}_X.pt")
            file_path = os.path.join(self.data_dir, f"{self.split}_X.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            X = torch.load(file_path, map_location=torch.device('cpu'))
            X = X[:, :, self.front_cut: self.tail_cut].float() #タイムステップのカット

            num_chunks = (X.size(0) + self.chunk_size - 1) // self.chunk_size

            for chunk_index in range(num_chunks): #メモリオーバー回避のためchunkに分けて前処理
                print(f"Preprocessing chunk {chunk_index}")
                start_index = chunk_index * self.chunk_size
                print(f"start_index {start_index}")
                end_index = min((chunk_index + 1) * self.chunk_size, X.size(0))
                print(f"end_index {end_index}")
                chunk = X[start_index:end_index,:,:]
                print(f"chunk_shape {chunk.shape}")
                processed_chunk = preprocess_chunk(chunk, self.new_fs, self.lowcut, self.highcut, self.baseline_correct, self.old_fs)
                print(f"processed_chunk_shape {processed_chunk.shape}")
                self.processed_X.append(processed_chunk)

            print(f"Loading {self.split}_subject_idxs.pt")
            file_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            self.subject_idxs = torch.load(file_path, map_location=torch.device('cpu'))

            if self.split in ["train", "val"]:
                print(f"Loading {self.split}_y.pt")
                file_path = os.path.join(self.data_dir, f"{self.split}_y.pt")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_path} does not exist.")
                self.y = torch.load(file_path, map_location=torch.device('cpu'))
                assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."
                if self.pretrain:
                    print(f"Loading {self.split}_image_paths.txt")
                    image_path_file = os.path.join(self.data_dir, f"{self.split}_image_paths.txt") #事前学習用画像ロード（結局使用できず）
                    if not os.path.exists(image_path_file):
                        raise FileNotFoundError(f"{image_path_file} does not exist.")
                    with open(image_path_file, 'r') as f:
                        self.image_paths = [line.strip() for line in f]
        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise

    def __len__(self) -> int:
        return sum([chunk.size(0) for chunk in self.processed_X])

    def __getitem__(self, i):
        chunk_index = i // self.chunk_size
        index_in_chunk = i % self.chunk_size
        chunk = self.processed_X[chunk_index]
        if hasattr(self, "y"):
            if self.pretrain:
                image_path = self.image_paths[i]
                if not os.path.dirname(image_path):
                    name_split = image_path.split('_')
                    dir_name = '_'.join(name_split[:-1])
                    image_path = os.path.join(self.image_dir, dir_name, image_path)
                else:
                    image_path = os.path.join(self.image_dir, image_path)
                image = self.preprocess_image(image_path)
                return chunk[index_in_chunk], image, self.y[i], self.subject_idxs[i]
            else:
                return chunk[index_in_chunk], self.y[i], self.subject_idxs[i]
        else:
            return chunk[index_in_chunk], self.subject_idxs[i]

    @staticmethod
    def preprocess_image(image_path):
        preprocess = Compose([
            Resize(224, interpolation=Image.BICUBIC),
            CenterCrop(224),
            ToTensor(),
            Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        ])
        image = Image.open(image_path).convert("RGB")
        return preprocess(image)

    @property
    def num_channels(self) -> int:
        return self.processed_X[0].size(1) if self.processed_X else 0

    @property
    def seq_len(self) -> int:
        return self.processed_X[0].size(2) if self.processed_X else 0
