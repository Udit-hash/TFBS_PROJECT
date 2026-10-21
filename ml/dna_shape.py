"""DNAshapeR-backed DNA-shape feature extraction.

The returned position axis is DNAshapeR's native output axis. It is not padded
or interpolated to match the nucleotide sequence length.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from config import PROCESSED_DATA_DIR
from preprocessing import clean_sequence


FEATURE_NAMES = ("MGW", "Roll", "ProT", "HelT")
PREDICTOR_NAME = "DNAshapeR"
R_SCRIPT = Path(__file__).resolve().with_name("dna_shape.R")


class DNAShapePredictionError(RuntimeError):
    """Raised when the configured DNAshapeR predictor cannot return features."""


@dataclass(frozen=True)
class DNAShapePrediction:
    """Single prediction padded to ``(4, max_native_positions)``."""

    values: np.ndarray
    native_lengths: tuple[int, ...]
    feature_names: tuple[str, ...]
    predictor: str
    predictor_version: str


@dataclass(frozen=True)
class DNAShapeBatch:
    """Feature batch padded to ``(sequences, 4, max_native_positions)``."""

    values: np.ndarray
    native_lengths: tuple[int, ...]
    feature_names: tuple[str, ...]
    predictor: str
    predictor_version: str


class DNAshapeRPredictor:
    """Run the locally installed DNAshapeR package through its Rscript CLI.

    DNAshapeR requires an R installation and the Bioconductor DNAshapeR
    package. Only canonical A/C/G/T sequences of the configured project length
    are accepted. Native output positions and any missing values are retained.
    """

    def __init__(
        self,
        cache_dir: str | Path | None = PROCESSED_DATA_DIR / "dna_shape_cache",
        rscript_executable: str | None = None,
        runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    ) -> None:
        configured_rscript = rscript_executable or os.environ.get(
            "DNA_SHAPER_RSCRIPT"
        )
        self.rscript = configured_rscript or shutil.which("Rscript")
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        self._runner = runner
        self._version: str | None = None

    @property
    def version(self) -> str:
        if self._version is None:
            if not self.rscript:
                raise DNAShapePredictionError(
                    "Rscript was not found. Install R and DNAshapeR, or set "
                    "DNA_SHAPER_RSCRIPT to the full path of Rscript."
                )
            try:
                result = self._runner(
                    [
                        self.rscript,
                        "--vanilla",
                        "-e",
                        (
                            "if (!requireNamespace('DNAshapeR', quietly=TRUE)) "
                            "quit(status=42); "
                            "cat(as.character(packageVersion('DNAshapeR')))"
                        ),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=60,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DNAShapePredictionError(
                    f"Could not query DNAshapeR through {self.rscript}: {exc}"
                ) from exc

            if result.returncode != 0 or not result.stdout.strip():
                detail = result.stderr.strip() or result.stdout.strip()
                raise DNAShapePredictionError(
                    "DNAshapeR is not available in the configured R installation. "
                    "Install it with BiocManager::install('DNAshapeR')."
                    + (f" R reported: {detail}" if detail else "")
                )
            self._version = result.stdout.strip()
        return self._version

    def predict(self, sequence: str) -> DNAShapePrediction:
        """Predict one sequence and retain each feature's native width."""
        batch = self.predict_batch([sequence])
        return DNAShapePrediction(
            values=batch.values[0],
            native_lengths=batch.native_lengths,
            feature_names=batch.feature_names,
            predictor=batch.predictor,
            predictor_version=batch.predictor_version,
        )

    def predict_batch(self, sequences: Sequence[str]) -> DNAShapeBatch:
        """Predict multiple sequences in one DNAshapeR call.

        Input ordering and duplicates are preserved. Optional cache entries are
        content-addressed by both sequence and DNAshapeR version.
        """
        if not sequences:
            raise ValueError("At least one DNA sequence is required.")

        cleaned = [clean_sequence(sequence) for sequence in sequences]
        version = self.version
        pending: list[str] = []
        results_by_sequence: dict[str, tuple[np.ndarray, tuple[int, ...]]] = {}

        for sequence in dict.fromkeys(cleaned):
            cached = self._read_cache(sequence, version)
            if cached is None:
                pending.append(sequence)
            else:
                results_by_sequence[sequence] = cached

        if pending:
            values, native_lengths = self._predict_uncached(pending)
            for index, sequence in enumerate(pending):
                result = values[index], native_lengths
                results_by_sequence[sequence] = result
                self._write_cache(sequence, version, *result)

        native_lengths_set = {
            result[1] for result in results_by_sequence.values()
        }
        if len(native_lengths_set) != 1:
            raise DNAShapePredictionError(
                "DNAshapeR returned inconsistent per-feature native position "
                "counts for equal-length input sequences."
            )
        native_lengths = next(iter(native_lengths_set))
        batch_values = np.stack(
            [results_by_sequence[sequence][0] for sequence in cleaned]
        ).astype(np.float32, copy=False)
        return DNAShapeBatch(
            values=batch_values,
            native_lengths=native_lengths,
            feature_names=FEATURE_NAMES,
            predictor=PREDICTOR_NAME,
            predictor_version=version,
        )

    def _predict_uncached(
        self, sequences: Sequence[str]
    ) -> tuple[np.ndarray, tuple[int, ...]]:
        if not self.rscript:
            raise DNAShapePredictionError(
                "Rscript was not found. Install R and DNAshapeR, or set "
                "DNA_SHAPER_RSCRIPT to the full path of Rscript."
            )
        if not R_SCRIPT.is_file():
            raise DNAShapePredictionError(
                f"DNAshapeR bridge script is missing: {R_SCRIPT}"
            )

        with tempfile.TemporaryDirectory(prefix="tfbs_dna_shape_") as tmp:
            temporary_dir = Path(tmp)
            fasta_path = temporary_dir / "input.fa"
            output_dir = temporary_dir / "output"
            with fasta_path.open("w", encoding="ascii", newline="\n") as fasta:
                for index, sequence in enumerate(sequences):
                    fasta.write(f">sequence_{index:08d}\n{sequence}\n")

            try:
                result = self._runner(
                    [
                        self.rscript,
                        "--vanilla",
                        str(R_SCRIPT),
                        str(fasta_path),
                        str(output_dir),
                        str(len(sequences)),
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise DNAShapePredictionError(
                    f"DNAshapeR prediction could not be completed: {exc}"
                ) from exc

            if result.returncode != 0:
                detail = result.stderr.strip() or result.stdout.strip()
                raise DNAShapePredictionError(
                    "DNAshapeR prediction failed."
                    + (f" R reported: {detail}" if detail else "")
                )

            feature_matrices = []
            for feature_name in FEATURE_NAMES:
                feature_path = output_dir / f"{feature_name}.tsv"
                if not feature_path.is_file():
                    raise DNAShapePredictionError(
                        f"DNAshapeR did not create its {feature_name} output."
                    )
                try:
                    matrix = np.loadtxt(
                        feature_path,
                        delimiter="\t",
                        dtype=np.float32,
                        ndmin=2,
                    )
                except (OSError, ValueError) as exc:
                    raise DNAShapePredictionError(
                        f"Could not parse DNAshapeR {feature_name} output: {exc}"
                    ) from exc
                if matrix.shape[0] != len(sequences) or matrix.shape[1] == 0:
                    raise DNAShapePredictionError(
                        f"DNAshapeR {feature_name} returned {matrix.shape}; "
                        f"expected {len(sequences)} rows and at least one position."
                    )
                if np.isinf(matrix).any():
                    raise DNAShapePredictionError(
                        f"DNAshapeR {feature_name} output contains infinite values."
                    )
                feature_matrices.append(matrix)

        native_lengths = tuple(matrix.shape[1] for matrix in feature_matrices)
        max_positions = max(native_lengths)
        padded_values = np.full(
            (len(sequences), len(FEATURE_NAMES), max_positions),
            np.nan,
            dtype=np.float32,
        )
        for feature_index, matrix in enumerate(feature_matrices):
            padded_values[:, feature_index, : matrix.shape[1]] = matrix
        return padded_values, native_lengths

    def _cache_path(self, sequence: str, version: str) -> Path | None:
        if self.cache_dir is None:
            return None
        digest = hashlib.sha256(
            f"cache-v2\0{PREDICTOR_NAME}\0{version}\0{sequence}".encode("ascii")
        ).hexdigest()
        return self.cache_dir / f"{digest}.npz"

    def _read_cache(
        self, sequence: str, version: str
    ) -> tuple[np.ndarray, tuple[int, ...]] | None:
        path = self._cache_path(sequence, version)
        if path is None or not path.is_file():
            return None
        try:
            with np.load(path, allow_pickle=False) as cache:
                values = cache["values"]
                native_lengths = tuple(cache["native_lengths"].tolist())
                names = tuple(cache["feature_names"].tolist())
                cached_sequence = str(cache["sequence"].item())
                cached_version = str(cache["version"].item())
        except (OSError, ValueError, KeyError) as exc:
            raise DNAShapePredictionError(
                f"Could not read DNAshapeR cache entry {path}: {exc}"
            ) from exc
        if (
            cached_sequence != sequence
            or cached_version != version
            or names != FEATURE_NAMES
            or values.ndim != 2
            or values.shape[0] != len(FEATURE_NAMES)
            or len(native_lengths) != len(FEATURE_NAMES)
            or min(native_lengths, default=0) < 1
            or values.shape[1] != max(native_lengths, default=0)
            or np.isinf(values).any()
            or any(
                not np.isnan(values[index, length:]).all()
                for index, length in enumerate(native_lengths)
            )
        ):
            raise DNAShapePredictionError(
                f"DNAshapeR cache entry failed validation: {path}"
            )
        return values.astype(np.float32, copy=False), native_lengths

    def _write_cache(
        self,
        sequence: str,
        version: str,
        values: np.ndarray,
        native_lengths: tuple[int, ...],
    ) -> None:
        path = self._cache_path(sequence, version)
        if path is None:
            return
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                dir=self.cache_dir,
                suffix=".npz",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                np.savez_compressed(
                    temporary_file,
                    values=values,
                    native_lengths=np.asarray(native_lengths, dtype=np.int64),
                    feature_names=np.asarray(FEATURE_NAMES),
                    sequence=np.asarray(sequence),
                    version=np.asarray(version),
                )
            os.replace(temporary_path, path)
        finally:
            if temporary_path is not None and temporary_path.exists():
                temporary_path.unlink()
