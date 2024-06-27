#CNN+LSTM

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
