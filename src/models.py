#CNN+LSTM
#特徴量データと被験者idを線形結合で組み込むHybridModelを追加

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class ConvLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 128) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 128),
            ConvBlock(128, 128)
        )

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            Rearrange("b d 1 -> b d"),
            nn.Linear(lstm_hidden_dim * 2, num_classes)  # bidirectional, so hidden_dim * 2
        )

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.conv_blocks(X)
        X = X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        X, _ = self.lstm(X)
        X = X.permute(0, 2, 1)  # Back to (batch, features, time)
        return self.head(X)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

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



class HybridModel(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 128, feature_dim: int = 4, feature_hid_dim: int = 16, feature_out_dim: int = 4) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 128),
            ConvBlock(128, 128)
        ) #時系列データを処理する畳み込み層（Convは下で定義）の関数

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True) #双方向LSTM

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, feature_hid_dim),
            nn.ReLU(),
            nn.Linear(feature_hid_dim, feature_out_dim)
        ) #特徴量データの線形結合層の関数

        self.head = nn.Sequential(
            nn.Linear(lstm_hidden_dim * 2 + feature_out_dim, num_classes)  # bidirectional, so hidden_dim * 2 + feature_out_dim
        ) #出力層の関数：1次元の入力テンソルに対して適応平均プーリング（最後だけだとあまり安定しない？）双方向LSTMと線形層の合計の次元をクラス数に変換

    def forward(self, X: torch.Tensor, f: torch.Tensor) -> torch.Tensor:

        X = self.conv_blocks(X)
        X = X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        X, _ = self.lstm(X)
        X = X.mean(dim=1)  # 時系列全体の平均を計算
        f = self.feature_fc(f)

        combine = self.head(torch.cat((X, f), dim=1))
        return combine


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

'''
class HybridModel(nn.Module):
    def __init__(self, num_classes: int, seq_len: int, in_channels: int, lstm_hidden_dim: int = 128, feature_dim: int = 10) -> None:
        super().__init__()

        self.conv_blocks = nn.Sequential(
            ConvBlock(in_channels, 128),
            ConvBlock(128, 128)
        ) #時系列データを処理する畳み込み層

        self.lstm = nn.LSTM(input_size=128, hidden_size=lstm_hidden_dim, batch_first=True, bidirectional=True)

        self.feature_fc = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128)
        ) #特徴量データの線形結合層

        self.head = nn.Linear(lstm_hidden_dim * 2 + 128, num_classes)  # bidirectional, so hidden_dim * 2 + feature_fc output

    def forward(self, X: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        X = self.conv_blocks(X) #時系列データを処理する畳み込み層
        X = X.permute(0, 2, 1)  # Prepare for LSTM: (batch, time, features)
        _, (X, _) = self.lstm(X)  # Get last hidden state from LSTM
        X = torch.cat((X[-2,:,:], X[-1,:,:]), dim=1)  # Concatenate the last hidden states from both directions

        f = self.feature_fc(f) #特徴量データの線形結合層

        combined = torch.cat((X, f), dim=1)
        return self.head(combined)

class ConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size: int = 3, p_drop: float = 0.1) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.conv0 = nn.Conv1d(in_dim, out_dim, kernel_size, padding="same")
        self.conv1 = nn.Conv1d(out_dim, out_dim, kernel_size, padding="same")

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
'''