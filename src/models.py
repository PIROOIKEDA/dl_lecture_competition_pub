#CNN+LSTM
#特徴量データと被験者idを線形結合で組み込むHybridModelを追加

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange

def moving_average(X, window_size=3):
    pad = (window_size - 1) // 2
    # Xの次元: (batch_size, num_channels, seq_len)
    X_padded = F.pad(X, (pad, pad), mode='reflect')  # パディングの適用: (batch_size, num_channels, seq_len + 2*pad)
    
    # weightの形状: (num_channels, 1, window_size)
    weight = torch.ones(X.size(1), 1, window_size, device=X.device) / window_size
    
    # 畳み込み操作
    X_avg = F.conv1d(X_padded, weight=weight, groups=X.size(1))
    return X_avg

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

class SimpleConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 64, window_size: int = 5) -> None:
        super().__init__()
         
        self.window_size = window_size

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16)
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(lstm_hidden_dim * 2, num_classes)  # bidirectional, so hidden_dim * 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X[:,:,35:] # EEGデータの最初の30stepを削除
        X = moving_average(X, self.window_size)
        X = self.conv_blocks(X)
        X = X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        X, _ = self.lstm(X)
        X = X.permute(0, 2, 1)  # Back to (batch, features, time)
        return self.head(X)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

class SimpleConvLSTMEncoder2(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 64, window_size: int = 25) -> None:
        super().__init__()

        self.window_size = window_size

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16)
        )

        self.lstm = nn.LSTM(input_size=16, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, num_classes)  # bidirectional, so hidden_dim * 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = X[:,:,35:] # EEGデータの最初の30stepを削除
        X = moving_average(X, self.window_size)
        X = self.conv_blocks(X)
        X = X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        _, (hn, _) = self.lstm(X)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate the final forward and backward hidden states
        return self.head(hn)


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.activation = nn.GELU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.activation(x)

class ParallelConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 64, window_size: int = 3):
        super(ParallelConvLSTMClassifier, self).__init__()
        self.window_size = window_size
        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 64),
            ConvBlock(64, 32),
            ConvBlock(32, 16)
        )
        self.lstm = nn.LSTM(input_size=in_channels, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, num_classes)  # bidirectional, so hidden_dim * 2
        )

    def forward(self, X):
        X = X[:,:,35:] # EEGデータの最初の30stepを削除
        print(X.shape)
        X = moving_average(X, self.window_size)
        print(X.shape)
        X = X.permute(0, 2, 1) 
        _, (hn, _) = self.lstm(X)
        hn = torch.cat((hn[-2], hn[-1]), dim=1)  # Concatenate the final forward and backward hidden states
        return self.head(hn)


class SubjectSpecificConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3, p_drop=0.3):
        super().__init__()
        padding = kernel_size // 2  # Manual calculation for "same" padding
        self.conv0 = nn.Conv1d(in_dim, 64, kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(64, 32, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(32, 16, kernel_size, padding=padding)
        self.conv3 = nn.Conv1d(16, 8, kernel_size, padding=padding)
        self.batchnorm0 = nn.BatchNorm1d(64)
        self.batchnorm1 = nn.BatchNorm1d(32)
        self.batchnorm2 = nn.BatchNorm1d(16)
        self.batchnorm3 = nn.BatchNorm1d(8)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, X):
        X = self.conv0(X)
        X = F.gelu(self.batchnorm0(X))
        X = self.conv1(X)
        X = F.gelu(self.batchnorm1(X))
        X = self.conv2(X)
        X = F.gelu(self.batchnorm2(X))
        X = self.conv3(X)
        X = F.gelu(self.batchnorm3(X))
        return self.dropout(X)

#被験者ごとに分ける
class ConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, out_channels: int=64, lstm_hidden_dim: int = 64, num_subjects=4, window_size: int = 35) -> None:
        super().__init__()

        self.window_size = window_size

        self.num_subjects = num_subjects

        # 各被験者ごとのサブネットワークを定義
        self.subject_specific_convs = nn.ModuleList([SubjectSpecificConvBlock(in_channels, out_channels) for _ in range(num_subjects)])



        self.lstm = nn.LSTM(8, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2, num_classes)  # bidirectional, so hidden_dim * 2
        )

    def forward(self, X: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        X = X[:,:,35:] # EEGデータの最初の30stepを削除
        X = moving_average(X, self.window_size)
        outputs = []
        idx_list = []

        for i in range(self.num_subjects):
            idx = (subject_id == i).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                subject_X = X[idx]  # X[idx] の形状は (batch, channels, seq_len)
                subject_output = self.subject_specific_convs[i](subject_X)
                outputs.append(subject_output)
                idx_list.append(idx)

        if len(outputs) > 0:
            combined_X = torch.cat(outputs, dim=0)  # Concatenate all subjects' outputs
            combined_idx = torch.cat(idx_list, dim=0)  # Concatenate all subjects' indices

            # 元の順番に戻す
            _, sorted_idx = torch.sort(combined_idx)
            combined_X = combined_X[sorted_idx]

            combined_X = combined_X.permute(0, 2, 1)  # Prepare for LSTM: (batch, channels, seq_len) -> (batch, seq_len, features)
            _, (hn, _) = self.lstm(combined_X)
            hn = torch.cat((hn[-2], hn[-1]), dim=1)  # 時系列全体の平均を計算
            return self.head(hn)
        else:
            # バッチ内にデータがない場合の対策
            batch_size = X.size(0)
            num_classes = self.head[0].out_features
            return torch.zeros((batch_size, num_classes), device=X.device)


        

#被験者ごとに分け、さらに特徴量を追加する
class HybridModel(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 128, feature_dim: int = 4, feature_hid_dim: int = 128, feature_out_dim: int = 128, num_subjects=4) -> None:

        super().__init__()
        self.num_subjects = num_subjects

        # 各被験者ごとのサブネットワークを定義
        self.subject_specific_convs = nn.ModuleList([SubjectSpecificConvBlock(in_channels, 128) for _ in range(num_subjects)])


        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True) #双方向LSTM

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_hid_dim),
            nn.ReLU(),
            nn.Linear(feature_hid_dim, feature_out_dim)
        ) #特徴量データの線形結合層の関数

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2 + feature_out_dim, num_classes)  # bidirectional, so hidden_dim * 2 + feature_out_dim
        ) #出力層の関数：1次元の入力テンソルに対して適応平均プーリング（最後だけだとあまり安定しない？）双方向LSTMと線形層の合計の次元をクラス数に変換

    def forward(self, X: torch.Tensor, f: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        # 被験者IDごとにデータを分割し、適切なサブネットワークに渡す
        outputs = []
        for i in range(self.num_subjects):
            idx = (subject_id == i).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                subject_X = X[idx]
                subject_output = self.subject_specific_convs[i](subject_X)
                outputs.append((subject_output, idx))

        # 出力を再結合してLSTMに渡す
        sorted_outputs = sorted(outputs, key=lambda x: x[1].min().item())
        combined_X = torch.cat([output[0] for output in sorted_outputs], dim=0)

        combined_X = combined_X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        combined_X, _ = self.lstm(combined_X)
        combined_X = combined_X.mean(dim=1)  # 時系列全体の平均を計算

        f = self.feature_fc(f)
        combined_features = torch.cat((combined_X, f), dim=1)
        return self.head(combined_features)


class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same") #in_dim, out_dim, kernel_size
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same") #out_dim, out_dim, kernel_size

        self.batchnorm0 = nn.BatchNorm1d(num_features=out_dim)
        self.batchnorm1 = nn.BatchNorm1d(num_features=out_dim)

        self.dropout = nn.Dropout(p_drop)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if self.in_dim == self.out_dim:
            X = self.conv0(X) + X  # skip connection
        else:
            X = self.conv0(X)

        X = F.gelu(self.batchnorm0(X))

        X = self.conv1(X) + X  # skip connection
        X = F.gelu(self.batchnorm1(X))

        return self.dropout(X)




