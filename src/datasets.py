#dataset
#フーリエ変換とARモデルによる特徴量を追加したEnhancedEEGDatasetを追加
#chunkに分けてロード



import os
import torch
import numpy as np
from scipy.signal import resample, butter, filtfilt, welch
from statsmodels.tsa.ar_model import AutoReg
from torch.utils.data import Dataset, DataLoader, Subset, Sampler, default_collate
import random

class CustomDataLoader(DataLoader):
    def __init__(self, dataset, num_subjects, batch_size, shuffle=True, **kwargs):
        super().__init__
        self.dataset = dataset
        self.num_subjects = num_subjects
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.subject_indices = {i: [] for i in range(num_subjects)}
        self._prepare_indices()
        print(f"Initialized CustomDataLoader with batch_size={self.batch_size} and num_subjects={self.num_subjects}")  # デバッグプリント文

    def _prepare_indices(self):
        for idx in range(len(self.dataset)):
            _, _, subject_id = self.dataset[idx]
            self.subject_indices[subject_id.item()].append(idx)

    def __iter__(self):
        if self.shuffle:
            for subject_id in self.subject_indices:
                random.shuffle(self.subject_indices[subject_id])

        min_samples = min(len(indices) for indices in self.subject_indices.values())
        print(f"Min samples per subject: {min_samples}")  # デバッグプリント文
        print(f"Batch size: {self.batch_size}, Num subjects: {self.num_subjects}")  # デバッグプリント文
        num_batches = min_samples // (self.batch_size // self.num_subjects)
        print(f"Number of batches: {num_batches}")  # デバッグプリント文

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



class EnhancedEEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", chunk_size=5000, new_fs=None, lowcut=None, highcut=None, baseline_correct=False, feature_extraction=False) -> None:
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.new_fs = new_fs
        self.lowcut = lowcut
        self.highcut = highcut
        self.baseline_correct = baseline_correct
        self.feature_extraction = feature_extraction
        self.num_subjects = 4
        self.num_classes = 1854

        # Load metadata
        self.load_metadata()

        # Check if preprocessed data exists, otherwise preprocess and save
        self.preprocessed_data_path = os.path.join(self.data_dir, f'{self.split}_data.pt') #splitごとに前処理データ保存
        self.preprocessed_features_path = os.path.join(self.data_dir, f'{self.split}_features.pt') #splitごとに特長量データ保存

        if not os.path.exists(self.preprocessed_data_path):
            self.preprocess_and_save_data()

        # Load preprocessed data
        self.load_preprocessed_data()

    def load_metadata(self):
        try:
            print(f"Loading {self.split}_X.pt metadata")
            file_path = os.path.join(self.data_dir, f"{self.split}_X.pt")
            if not os.path.exists(file_path): #エラーチェック
                raise FileNotFoundError(f"{file_path} does not exist。") #エラーチェック
            self.X_metadata = torch.load(file_path)

            print(f"Loading {self.split}_subject_idxs.pt") #エラーチェック
            file_path = os.path.join(self.data_dir, f"{self.split}_subject_idxs.pt") 
            if not os.path.exists(file_path): #エラーチェック
                raise FileNotFoundError(f"{file_path} does not exist。") #エラーチェック
            self.subject_idxs = torch.load(file_path)

            if self.split in ["train", "val"]:
                print(f"Loading {self.split}_y.pt")
                file_path = os.path.join(self.data_dir, f"{self.split}_y.pt")
                if not os.path.exists(file_path): #エラーチェック 
                    raise FileNotFoundError(f"{file_path} does not exist。") #エラーチェック
                self.y = torch.load(file_path)

        except Exception as e:
            print(f"An error occurred while loading metadata: {e}") #エラーチェック
            raise

        print(f"Loaded metadata for dataset: {self.split} with {len(self.X_metadata)} samples。") #エラーチェック

    def preprocess(self, X_np):
        if self.baseline_correct:
            baseline = np.mean(X_np, axis=-1, keepdims=True)
            X_np = X_np - baseline 

        if self.new_fs is not None:
            num_samples = int(X_np.shape[-1] * (self.new_fs / 1000))
            X_np = resample(X_np, num_samples, axis=-1)

        if self.lowcut is not None:
            b, a = butter(5, self.lowcut / (0.5 * self.new_fs), btype='high')
            X_np = filtfilt(b, a, X_np, axis=-1)

        if self.highcut is not None:
            b, a = butter(5, self.highcut / (0.5 * self.new_fs), btype='low')
            X_np = filtfilt(b, a, X_np, axis=-1)

        return X_np

    def preprocess_and_save_data(self):
        data_list = []
        features_list = []
        subject_list = []

        for start_idx in range(0, len(self.X_metadata), self.chunk_size):
            end_idx = min(start_idx + self.chunk_size, len(self.X_metadata))
            chunk_data = [self.preprocess(self.X_metadata[i].numpy()) for i in range(start_idx, end_idx)]
            chunk_data = np.array(chunk_data)  # chunkごとにnp.array型に
            data_list.append(chunk_data)

            if self.feature_extraction:
                chunk_features = self.extract_features(chunk_data, self.new_fs)  # フーリエ変換、ARモデル適用
                features_list.append(chunk_features)  # chunkごとにfeatures_listに追加

            subject_list.append(self.subject_idxs[start_idx:end_idx])

        data = torch.tensor(np.concatenate(data_list, axis=0), dtype=torch.float32)  # データを結合
        torch.save(data, self.preprocessed_data_path)  # 前処理データを保存

        if self.feature_extraction:
            features = torch.tensor(np.concatenate(features_list, axis=0), dtype=torch.float32)
            # 特徴量データにsubject_idxを結合
            subjects = torch.tensor(np.concatenate(subject_list, axis=0), dtype=torch.long)
            features_with_subjects = torch.cat((features, self.subject_idxs.float()), dim=1)
            torch.save(features_with_subjects, self.preprocessed_features_path) #特長量データ保存

    def load_preprocessed_data(self):
        self.data = torch.load(self.preprocessed_data_path)
        if self.feature_extraction:
            self.features = torch.load(self.preprocessed_features_path)

    def extract_features(self, eeg_signals, fs, bands=[(8, 13), (13, 30), (30, 45)], lags=5):
        def extract_band_power(eeg_signal, fs, band):
            f, Pxx = welch(eeg_signal, fs=fs, nperseg=256)
            band_power = np.sum(Pxx[(f >= band[0]) & (f <= band[1])])
            return band_power

        def extract_ar_features(eeg_signal, lags=5):
            model = AutoReg(eeg_signal, lags=lags)
            model_fit = model.fit()
            ar_params = model_fit.params
            return ar_params

        all_features = []
        for signal in eeg_signals:
            features = []
            for band in bands:
                power = extract_band_power(signal, fs, band)
                features.append(power)
            ar_features = extract_ar_features(signal, lags)
            features.extend(ar_features)
            all_features.append(features)
        return np.array(all_features)

    def one_hot_encode(self, idx, num_classes):
        one_hot = torch.zeros(num_classes)
        one_hot[idx] = 1.0
        return one_hot

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, i):
        X = self.data[i]
        subject_id = self.subject_idxs[i]
        subject_one_hot = self.one_hot_encode(subject_id, self.num_subjects)
        if self.feature_extraction:
            f = self.features[i]
        else:
            f = subject_one_hot
        if hasattr(self, "y"):
            y = self.y[i]
            return X, f, y
        else:
            return X, f

    @property
    def num_channels(self) -> int:
        return self.data.shape[1]

    @property
    def seq_len(self) -> int:
        return self.data.shape[2]

class ThingsMEGDataset(Dataset):
    def __init__(self, split: str, data_dir: str = "data", new_fs=None, lowcut=None, highcut=None, baseline_correct=False) -> None:
        """
        Initialize dataset and load data.
        """
        super().__init__()

        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        self.split = split
        self.num_classes = 1854

        # 各ファイルの読み込みにプリント文を追加
        # Load the data files
        try:
            print(f"Loading {split}_X.pt")
            file_path = os.path.join(data_dir, f"{split}_X.pt")
            print(f"Checking file existence: {os.path.exists(file_path)}")  # ファイルの存在確認
            print(f"File size: {os.path.getsize(file_path) / (1024 * 1024)} MB")  # ファイルのサイズ確認

            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            self.X = torch.load(file_path)
            print(f"Loaded {split}_X.pt")

            print(f"Loading {split}_subject_idxs.pt")
            self.subject_idxs = torch.load(os.path.join(data_dir, f"{split}_subject_idxs.pt"))
            print(f"Loaded {split}_subject_idxs.pt")
            file_path = os.path.join(data_dir, f"{split}_subject_idxs.pt")
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"{file_path} does not exist.")
            self.subject_idxs = torch.load(file_path)

            if split in ["train", "val"]:
                print(f"Loading {split}_y.pt")
                self.y = torch.load(os.path.join(data_dir, f"{split}_y.pt"))
                print(f"Loaded {split}_y.pt")
                file_path = os.path.join(data_dir, f"{split}_y.pt")
                if not os.path.exists(file_path):
                    raise FileNotFoundError(f"{file_path} does not exist.")
                self.y = torch.load(file_path)
                assert len(torch.unique(self.y)) == self.num_classes, "Number of classes do not match."

        except Exception as e:
            print(f"An error occurred while loading data: {e}")
            raise

        print(f"Loaded dataset: {split} with {len(self.X)} samples.")

    def preprocess(self, new_fs=None, lowcut=None, highcut=None, baseline_correct=False, old_fs=200):
        """
        Preprocess the data.
        """
        X_np = self.X.numpy()

        if baseline_correct:
            print("Applying baseline correction...")
            baseline = np.mean(X_np, axis=-1, keepdims=True)
            X_np = X_np - baseline

        if new_fs is not None and new_fs != old_fs:
            print(f"Resampling from {old_fs}Hz to {new_fs}Hz...")
            num_samples = int(X_np.shape[-1] * (new_fs / old_fs))
            X_np = resample(X_np, num_samples, axis=-1)

        if lowcut is not None:
            print(f"Applying high-pass filter with cutoff {lowcut}Hz...")
            nyquist = 0.5 * new_fs if new_fs is not None else 0.5 * old_fs
            low = lowcut / nyquist
            b, a = butter(5, low, btype='high')
            X_np = filtfilt(b, a, X_np, axis=-1)

        if highcut is not None:
            print(f"Applying low-pass filter with cutoff {highcut}Hz...")
            nyquist = 0.5 * new_fs if new_fs is not None else 0.5 * old_fs
            high = highcut / nyquist
            b, a = butter(5, high, btype='low')
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



