"""
Rebuild the ENCODE ChIP-seq benchmark datasets (Zeng et al. 2016, used by DeepSTF / MLSNet)
directly from the original ENCODE uniform TFBS peak files.

Why: the original download site (cnn.csail.mit.edu/motif_discovery) is offline, but the
peak files it was made from are still hosted by UCSC. This script repeats the same recipe:

  positives = 101 bp windows centred on each ChIP-seq peak summit (hg19)
  negatives = dinucleotide-shuffled copies of the positives (same composition, motif destroyed)

Output per dataset (same row format as the original benchmark):
  data/encode/<dataset>/train.data   ">chr4:18024376-18024477 ACGT... 1"
  data/encode/<dataset>/test.data

Run it in Google Colab (it downloads the ~800 MB hg19 genome once):
  python build_encode_dataset.py --names wgEncodeAwgTfbsSydhK562CtcfbIggrabUniPk
  python build_encode_dataset.py --list-file datasets_165.txt --limit 10
"""

from __future__ import annotations

import argparse
import gzip
import random
import shutil
import urllib.request
from pathlib import Path

from data_utils import TwoBitReader, dinucleotide_shuffle, write_benchmark

HERE = Path(__file__).resolve().parent
PEAK_MIRRORS = [
    "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/{name}.narrowPeak.gz",
    "https://hgdownload.cse.ucsc.edu/goldenPath/hg19/encodeDCC/wgEncodeAwgTfbsUniform/{name}.narrowPeak.gz",
]
GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.2bit"
HALF = 50  # 50 + 1 + 50 = 101 bp


def download(urls: list[str], dest: Path) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            print(f"  downloading {url}")
            with urllib.request.urlopen(url, timeout=120) as resp, open(dest, "wb") as out:
                shutil.copyfileobj(resp, out, length=1 << 20)
            return dest
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{url}: {exc}")
            dest.unlink(missing_ok=True)
    raise RuntimeError("Download failed:\n  " + "\n  ".join(errors))


def read_peaks(path: Path) -> list[tuple]:
    """narrowPeak: chrom start end name score strand signal p q summit_offset"""
    peaks = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            f = ln.rstrip("\n").split("\t")
            if len(f) < 10 or f[0].startswith(("track", "#")):
                continue
            chrom, start, end = f[0], int(f[1]), int(f[2])
            signal, summit = float(f[6]), int(f[9])
            centre = start + summit if summit >= 0 else (start + end) // 2
            peaks.append((signal, chrom, centre))
    peaks.sort(key=lambda p: -p[0])
    return peaks


def build_one(name: str, genome: TwoBitReader, out_root: Path, max_peaks: int, test_frac: float,
              seed: int, cache: Path) -> dict:
    peak_file = download([u.format(name=name) for u in PEAK_MIRRORS], cache / f"{name}.narrowPeak.gz")
    peaks = read_peaks(peak_file)

    positives, seen = [], set()
    for _, chrom, centre in peaks:
        if chrom not in genome.index:
            continue
        s, e = centre - HALF, centre + HALF + 1
        if s < 0 or e > genome.chrom_size(chrom):
            continue
        seq = genome.fetch(chrom, s, e)
        if "N" in seq or seq in seen:
            continue
        seen.add(seq)
        positives.append((f"{chrom}:{s}-{e}", seq))
        if len(positives) >= max_peaks:
            break

    rng = random.Random(seed)
    rng.shuffle(positives)
    n_test = int(round(len(positives) * test_frac))
    splits = {"test": positives[:n_test], "train": positives[n_test:]}

    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {"dataset": name, "peaks_in_file": len(peaks)}
    for split, pos in splits.items():
        ids, seqs, labels = [], [], []
        for pid, seq in pos:
            # each positive and its own shuffled negative stay in the same split (no leakage)
            ids += [pid, pid + "_shuf"]
            seqs += [seq, dinucleotide_shuffle(seq, rng)]
            labels += [1, 0]
        write_benchmark(out_dir / f"{split}.data", ids, seqs, labels)
        stats[split] = len(seqs)
    print(f"  {name}: train={stats['train']} test={stats['test']} (from {len(peaks)} peaks)")
    return stats


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--names", nargs="*", default=[], help="dataset names, e.g. wgEncodeAwgTfbsSydhK562CtcfbIggrabUniPk")
    ap.add_argument("--list-file", help="text file with one dataset name per line (e.g. datasets_165.txt)")
    ap.add_argument("--limit", type=int, default=0, help="only the first N names from --list-file")
    ap.add_argument("--genome", default=str(HERE / "data" / "genome" / "hg19.2bit"))
    ap.add_argument("--out", default=str(HERE / "data" / "encode"))
    ap.add_argument("--max-peaks", type=int, default=10000, help="strongest peaks to keep per dataset")
    ap.add_argument("--test-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    names = list(args.names)
    if args.list_file:
        listed = [ln.strip() for ln in open(args.list_file) if ln.strip() and not ln.startswith("#")]
        names += listed[: args.limit] if args.limit else listed
    if not names:
        ap.error("give --names or --list-file")

    genome_path = download([GENOME_URL], Path(args.genome))
    genome = TwoBitReader(genome_path)
    cache = Path(args.out) / "_peaks"
    for name in names:
        try:
            build_one(name, genome, Path(args.out), args.max_peaks, args.test_frac, args.seed, cache)
        except Exception as exc:  # noqa: BLE001
            print(f"  !! skipped {name}: {exc}")


if __name__ == "__main__":
    main()
