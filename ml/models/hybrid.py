import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    """Injects positional awareness so self-attention understands sequence order."""
    def __init__(self, d_model, max_len=200):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1)]

class TFBS_TriBranch(nn.Module):
    """
    Full Tri-Branch Architecture from Section IV-B of Research Paper:
    - Multi-scale 1D-CNN (8, 16, 24 bp filters)
    - Bidirectional LSTM
    - Multi-Head Self-Attention Transformer
    """
    def __init__(self, seq_len=101, lstm_hidden=32, nhead=4, dropout=0.2):
        super(TFBS_TriBranch, self).__init__()
        
        # 1. Multi-Scale Convolutional Feature Extractors (8, 16, 24 bp)
        self.conv8 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=8, padding=4)
        self.conv16 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=16, padding=8)
        self.conv24 = nn.Conv1d(in_channels=4, out_channels=16, kernel_size=24, padding=12)
        
        # 16 + 16 + 16 = 48 feature channels
        total_conv_channels = 48
        self.bn_conv = nn.BatchNorm1d(total_conv_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)  # 101 -> 50 positions
        self.drop = nn.Dropout(dropout)
        
        # 2. Sequential BiLSTM Branch
        self.bilstm = nn.LSTM(
            input_size=total_conv_channels,
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        
        # 3. Global Positional Transformer Self-Attention Branch
        self.pos_encoder = PositionalEncoding(d_model=total_conv_channels, max_len=150)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=total_conv_channels,
            nhead=nhead,
            dim_feedforward=64,
            dropout=dropout,
            batch_first=True,
            norm_first=True  # Pre-LN ensures stable training convergence
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Concatenated dimensions:
        # CNN MaxPool: 48
        # BiLSTM MaxPool: lstm_hidden * 2 = 64
        # Transformer MaxPool: 48
        fused_dim = total_conv_channels + (lstm_hidden * 2) + total_conv_channels  # 160
        
        # 4. Dense Classification Head
        self.classifier = nn.Sequential(
            nn.Linear(fused_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x: (batch, 4, seq_len)
        
        # Branch 1: Multi-scale Convolutions
        c8 = F.relu(self.conv8(x))
        c16 = F.relu(self.conv16(x))
        c24 = F.relu(self.conv24(x))
        
        # Align target length
        min_len = min(c8.size(2), c16.size(2), c24.size(2))
        conv_feats = torch.cat([c8[:, :, :min_len], c16[:, :, :min_len], c24[:, :, :min_len]], dim=1)
        
        conv_feats = self.bn_conv(conv_feats)
        pooled_seq = self.drop(self.pool(conv_feats))  # (batch, 48, reduced_len)
        
        # CNN Global Max Representation
        cnn_rep, _ = torch.max(pooled_seq, dim=2)  # (batch, 48)
        
        # Format for sequential branches: (batch, seq_len_reduced, 48)
        seq_tokens = pooled_seq.permute(0, 2, 1)
        
        # Branch 2: BiLSTM representation
        lstm_out, _ = self.bilstm(seq_tokens)
        lstm_rep, _ = torch.max(lstm_out, dim=1)    # (batch, 64)
        
        # Branch 3: Transformer with Positional Encoding
        trans_tokens = self.pos_encoder(seq_tokens)
        trans_out = self.transformer(trans_tokens)
        trans_rep, _ = torch.max(trans_out, dim=1)  # (batch, 48)
        
        # Tri-Branch Concatenation
        fused = torch.cat([cnn_rep, lstm_rep, trans_rep], dim=1)  # (batch, 160)
        
        logits = self.classifier(fused)  # Returns (batch, 1) to match BCE targets exactly
        return logits