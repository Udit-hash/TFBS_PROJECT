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
    def __init__(
        self,
        seq_len=101,
        lstm_hidden=48,
        nhead=4,
        dropout=0.25,
        conv_channels=24,
        classifier_hidden=128,
        classifier_intermediate=True,
        classifier_output=64,
    ):
        super(TFBS_TriBranch, self).__init__()
        
        # 1. Multi-Scale Convolutional Feature Extractors (8, 16, 24 bp)
        self.conv8 = nn.Conv1d(in_channels=4, out_channels=conv_channels, kernel_size=8, padding=4)
        self.conv16 = nn.Conv1d(in_channels=4, out_channels=conv_channels, kernel_size=16, padding=8)
        self.conv24 = nn.Conv1d(in_channels=4, out_channels=conv_channels, kernel_size=24, padding=12)
        
        total_conv_channels = conv_channels * 3
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
        # CNN MaxPool: 72
        # BiLSTM MaxPool: lstm_hidden * 2 = 96
        # Transformer MaxPool: 72
        fused_dim = total_conv_channels + (lstm_hidden * 2) + total_conv_channels

        # 4. Dense Classification Head
        classifier_layers = [
            nn.Linear(fused_dim, classifier_hidden),
            nn.GELU(),
            nn.BatchNorm1d(classifier_hidden),
            nn.Dropout(dropout),
        ]
        if classifier_intermediate:
            classifier_layers.extend([
                nn.Linear(classifier_hidden, classifier_output),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            classifier_hidden = classifier_output
        classifier_layers.append(nn.Linear(classifier_hidden, 1))
        self.classifier = nn.Sequential(*classifier_layers)

    def encode_sequence(self, x):
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
        
        # Tri-Branch sequence representation: (batch, 240)
        return torch.cat([cnn_rep, lstm_rep, trans_rep], dim=1)
        
    def forward(self, x):
        return self.classifier(self.encode_sequence(x))

class TFBS_TriBranchV2(nn.Module):
    """
    Tri-branch model, version 2: same idea as TFBS_TriBranch (multi-scale CNN feeding a BiLSTM
    and a Transformer), but with a deeper shared convolution trunk, because version 1's single
    thin convolution layer limited it (it scored below the CNN baselines on FOSL2).

      multi-scale conv (8/16/24 bp, `filters` each) -> BN -> ReLU -> max-pool 2
      conv 5 bp (`trunk_channels`)                  -> BN -> ReLU -> max-pool 2   (~25 positions)
      branch 1: global max over the trunk (CNN summary)
      branch 2: BiLSTM over the 25 positions        (switch off with use_lstm=False)
      branch 3: Transformer encoder over them       (switch off with use_transformer=False)
      concatenate -> small classifier -> 1 logit

    The switches give the ablation models used in the paper's comparison.
    """

    def __init__(self, seq_len=101, filters=32, trunk_channels=128, lstm_hidden=64, nhead=4,
                 ff_dim=256, dropout=0.3, use_lstm=True, use_transformer=True):
        super().__init__()
        self.use_lstm = use_lstm
        self.use_transformer = use_transformer
        self.conv8 = nn.Conv1d(4, filters, kernel_size=8, padding=4)
        self.conv16 = nn.Conv1d(4, filters, kernel_size=16, padding=8)
        self.conv24 = nn.Conv1d(4, filters, kernel_size=24, padding=12)
        self.bn1 = nn.BatchNorm1d(filters * 3)
        self.conv2 = nn.Conv1d(filters * 3, trunk_channels, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(trunk_channels)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.drop = nn.Dropout(dropout)

        fused = trunk_channels
        if use_lstm:
            self.bilstm = nn.LSTM(trunk_channels, lstm_hidden, batch_first=True, bidirectional=True)
            fused += 2 * lstm_hidden
        if use_transformer:
            self.pos_encoder = PositionalEncoding(d_model=trunk_channels, max_len=150)
            layer = nn.TransformerEncoderLayer(d_model=trunk_channels, nhead=nhead, dim_feedforward=ff_dim,
                                               dropout=dropout, batch_first=True, norm_first=True)
            self.transformer = nn.TransformerEncoder(layer, num_layers=1)
            fused += trunk_channels
        self.classifier = nn.Sequential(
            nn.Linear(fused, 128), nn.GELU(), nn.Dropout(dropout), nn.Linear(128, 1)
        )

    def encode_sequence(self, x):
        c8, c16, c24 = self.conv8(x), self.conv16(x), self.conv24(x)
        n = min(c8.size(2), c16.size(2), c24.size(2))
        h = torch.cat([c8[:, :, :n], c16[:, :, :n], c24[:, :, :n]], dim=1)
        h = self.pool(F.relu(self.bn1(h)))
        h = self.drop(self.pool(F.relu(self.bn2(self.conv2(h)))))    # (B, C, ~25)
        reps = [h.amax(dim=2)]
        tokens = h.permute(0, 2, 1)                                    # (B, ~25, C)
        if self.use_lstm:
            out, _ = self.bilstm(tokens)
            reps.append(out.amax(dim=1))
        if self.use_transformer:
            reps.append(self.transformer(self.pos_encoder(tokens)).amax(dim=1))
        return torch.cat(reps, dim=1)

    def forward(self, x):
        return self.classifier(self.encode_sequence(x))
