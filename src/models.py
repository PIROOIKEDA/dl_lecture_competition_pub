import torch
import torch.nn as nn
import torch.nn.functional as F




class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=64):
        super(ChannelAttention, self).__init__()
        self.reduced_channels = max(1, in_channels // reduction_ratio)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(
            nn.Conv1d(in_channels, self.reduced_channels, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(self.reduced_channels, in_channels, kernel_size=1, padding=0)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SubjectSpecificConvBlock(nn.Module):
    def __init__(self, in_dim, out_dim=64, kernel_size=15, p_drop=0.3):
        super().__init__()
        padding = kernel_size // 2  # Manual calculation for "same" padding
        self.conv0 = nn.Conv1d(in_dim, 128, kernel_size, padding=padding)
        self.conv1 = nn.Conv1d(128, out_dim, kernel_size, padding=padding)
        self.conv2 = nn.Conv1d(out_dim, out_dim, kernel_size, padding=padding)
        self.batchnorm0 = nn.BatchNorm1d(128)
        self.batchnorm1 = nn.BatchNorm1d(out_dim)
        self.batchnorm2 = nn.BatchNorm1d(out_dim)
        self.dropout = nn.Dropout(p_drop)
        self.channel_attention0 = ChannelAttention(128)
        self.channel_attention1 = ChannelAttention(out_dim)
        self.channel_attention2 = ChannelAttention(out_dim)

    def forward(self, x):
        x = self.conv0(x)
        x = F.gelu(self.batchnorm0(x))
        x = self.channel_attention0(x) * x
        x = self.conv1(x)
        x = F.gelu(self.batchnorm1(x))
        x = self.channel_attention1(x) * x
        x = self.conv2(x)
        x = F.gelu(self.batchnorm2(x))
        x = self.channel_attention2(x) * x
        return self.dropout(x)


class AttentionPooling(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.query = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (seq_len, batch, d_model)
        scores = self.query(x)  # (seq_len, batch, 1)
        weights = F.softmax(scores, dim=0)  # (seq_len, batch, 1)
        weighted_sum = (weights * x).sum(dim=0)  # (batch, d_model)
        return weighted_sum


class ConvTransformerClassifier(nn.Module):
    def __init__(self, out_dim: int, seq_len: int, in_channels: int, out_channels: int = 8, transformer_hidden_dim: int = 128, num_subjects: int =4,num_heads: int = 8, num_layers: int = 2) -> None:
        super().__init__()


        self.num_subjects = num_subjects

        self.subject_specific_convs = nn.ModuleList([SubjectSpecificConvBlock(in_channels, out_channels) for _ in range(num_subjects)])

        encoder_layer = nn.TransformerEncoderLayer(d_model=out_channels, nhead=num_heads, dim_feedforward=transformer_hidden_dim)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.attention_pooling = AttentionPooling(out_channels)
        self.head = nn.Sequential(
            nn.Linear(out_channels, out_dim)
        )

    def forward(self, X: torch.Tensor, subject_id: torch.Tensor) -> torch.Tensor:
        outputs = []
        idx_list = []

        for i in range(self.num_subjects):
            idx = (subject_id == i).nonzero(as_tuple=True)[0]
            if len(idx) > 0:
                subject_X = X[idx]
                subject_output = self.subject_specific_convs[i](subject_X)
                outputs.append(subject_output)
                idx_list.append(idx)

        if len(outputs) > 0:
            combined_X = torch.cat(outputs, dim=0)
            combined_idx = torch.cat(idx_list, dim=0)
            _, sorted_idx = torch.sort(combined_idx)
            combined_X = combined_X[sorted_idx]

            combined_X = combined_X.permute(2, 0, 1)  # (batch, channels, seq_len) -> (seq_len, batch, features)
            transformer_output = self.transformer(combined_X)
            pooled_output = self.attention_pooling(transformer_output)
            return self.head(pooled_output)
        else:
            batch_size = X.size(0)
            out_dim = self.head[0].out_features
            return torch.zeros((batch_size, out_dim), device=X.device)
