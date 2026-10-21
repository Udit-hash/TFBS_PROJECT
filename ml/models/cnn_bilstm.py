import torch
import torch.nn as nn


class TFBS_CNN_BiLSTM(nn.Module):
    """CNN motif encoder followed by a bidirectional recurrent encoder."""

    def __init__(self, lstm_hidden: int = 64, dropout: float = 0.3):
        super().__init__()
        self.convolutions = nn.Sequential(
            nn.Conv1d(4, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(64, 96, kernel_size=5, padding=2),
            nn.BatchNorm1d(96),
            nn.ReLU(),
        )
        self.bilstm = nn.LSTM(
            input_size=96,
            hidden_size=lstm_hidden,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.convolutions(x).transpose(1, 2)
        recurrent, _ = self.bilstm(features)
        representation = recurrent.amax(dim=1)
        return self.classifier(representation)
