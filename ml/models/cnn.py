import torch
import torch.nn as nn


class TFBS_CNN(nn.Module):

    def __init__(self):

        super().__init__()

        # ====================================================
        # CONVOLUTIONAL FEATURE EXTRACTION
        # ====================================================

        self.conv_layers = nn.Sequential(

            # ------------------------------------------------
            # Layer 1
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=4,
                out_channels=64,
                kernel_size=8,
                padding=4
            ),

            nn.BatchNorm1d(64),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Layer 2
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=8,
                padding=4
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(),

            nn.MaxPool1d(
                kernel_size=2
            ),

            # ------------------------------------------------
            # Layer 3
            # ------------------------------------------------

            nn.Conv1d(
                in_channels=128,
                out_channels=256,
                kernel_size=8,
                padding=4
            ),

            nn.BatchNorm1d(256),

            nn.ReLU()
        )

        # ====================================================
        # GLOBAL MAX POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveMaxPool1d(1)

        # ====================================================
        # CLASSIFICATION HEAD
        # ====================================================

        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(
                0.3
            ),

            nn.Linear(
                64,
                1
            )
        )

    def forward(self, x):

        # Input:
        #
        # [batch_size, 4, 101]

        x = self.conv_layers(x)

        # Example:
        #
        # [batch_size, 256, sequence_length]

        x = self.global_pool(x)

        # [batch_size, 256, 1]

        x = self.classifier(x)

        # [batch_size, 1]

        return x