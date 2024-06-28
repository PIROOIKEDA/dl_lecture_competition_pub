#dataset
#フーリエ変換とARモデルによる特徴量を追加したEnhancedEEGDatasetを追加
#chunkに分けてロード

import os
import torch
import numpy as np
from scipy.signal import resample, butter, filtfilt, welch
from statsmodels.tsa.ar_model import AutoReg
from torch.utils.data import Dataset

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
