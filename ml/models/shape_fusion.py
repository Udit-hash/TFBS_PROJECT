import torch
import torch.nn as nn

from models.hybrid import TFBS_TriBranch


class TFBS_TriBranchShape(nn.Module):
    """Fuse the tri-branch sequence representation with DNAshapeR features.

    Inputs:
      sequence: [batch, 4, nucleotide_length]
      shape:    [batch, 4, max_native_positions], right-padded with NaN

    Each feature is encoded on its own native position axis, so base-pair
    and base-pair-step features are not assumed to share aligned positions.
    Missing or batch-padding values remain NaN until converted to standardized
    zeros and accompanied by a validity mask.
    """

    def __init__(
        self,
        seq_len: int = 101,
        shape_mean: torch.Tensor | None = None,
        shape_std: torch.Tensor | None = None,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.sequence_length = seq_len
        self.sequence_model = TFBS_TriBranch(seq_len=seq_len)
        mean = torch.zeros(4) if shape_mean is None else torch.as_tensor(shape_mean)
        std = torch.ones(4) if shape_std is None else torch.as_tensor(shape_std)
        if mean.shape != (4,) or std.shape != (4,):
            raise ValueError("DNA-shape normalization statistics must have 4 values.")
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            raise ValueError("DNA-shape normalization statistics must be finite.")
        if (std <= 0).any():
            raise ValueError("DNA-shape standard deviations must be positive.")
        self.register_buffer("shape_mean", mean.float())
        self.register_buffer("shape_std", std.float())

        self.shape_encoder = nn.Sequential(
            nn.Conv1d(2, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, 16, kernel_size=5, padding=2),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(240 + 64, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def encode(self, sequence: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != 4:
            raise ValueError("Sequence input must have shape [batch, 4, length].")
        if sequence.shape[2] != self.sequence_length:
            raise ValueError(
                f"Sequence input must have length {self.sequence_length}."
            )
        if shape.ndim != 3 or shape.shape[1] != 4 or shape.shape[2] < 1:
            raise ValueError(
                "DNA-shape input must have shape [batch, 4, native_positions]."
            )
        if sequence.shape[0] != shape.shape[0]:
            raise ValueError("Sequence and DNA-shape batch sizes must match.")
        if torch.isinf(shape).any():
            raise ValueError("DNA-shape input cannot contain infinite values.")

        valid = torch.isfinite(shape)
        safe_shape = torch.where(valid, shape, self.shape_mean[None, :, None])
        normalized_shape = (
            safe_shape - self.shape_mean[None, :, None]
        ) / self.shape_std[None, :, None]
        normalized_shape = torch.where(
            valid, normalized_shape, torch.zeros_like(normalized_shape)
        )
        shape_input = torch.stack(
            (normalized_shape, valid.to(shape.dtype)), dim=2
        ).reshape(
            sequence.shape[0] * 4, 2, shape.shape[2]
        )
        shape_representation = self.shape_encoder(shape_input).reshape(
            sequence.shape[0], 64
        )
        sequence_representation = self.sequence_model.encode_sequence(sequence)
        return torch.cat((sequence_representation, shape_representation), dim=1)

    def forward(self, sequence: torch.Tensor, shape: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(sequence, shape))
