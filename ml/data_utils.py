"""
Data utilities for the TFBS project (no deep-learning libraries needed).

- read_sequences(): loads any of the formats we use
    * ENCODE / Zeng et al. 2016 benchmark rows:  ">chr4:18024376-18024476 ACGT... 1"
      (space- or tab-separated, as in the 690/165-dataset benchmark)
    * CSV with a header:                          chr_id,sequence,label
    * FASTA of bound sequences (label 1); negatives are then made by dinucleotide shuffling
- dinucleotide_shuffle(): Altschul–Erickson shuffle that keeps every dinucleotide count,
  the standard way to make matched negative controls (same as MEME's fasta-dinucleotide-shuffle)
- TwoBitReader: minimal reader for UCSC .2bit genomes (used to cut 101 bp windows from hg19)
"""

from __future__ import annotations

import csv
import random
import struct
from collections import Counter
from pathlib import Path

import numpy as np

VALID = set("ACGT")


# ============================================================
# DINUCLEOTIDE SHUFFLE (Altschul & Erickson, 1985)
# ============================================================

def _connected_to_last(last_edges: dict, last_char: str, letters: set) -> bool:
    """Every letter must reach `last_char` through the chosen last-exit edges."""
    reach = {last_char}
    changed = True
    while changed:
        changed = False
        for x, y in last_edges.items():
            if x not in reach and y in reach:
                reach.add(x)
                changed = True
    return reach >= letters


def dinucleotide_shuffle(seq: str, rng: random.Random) -> str:
    """Random sequence with exactly the same dinucleotide counts, first and last letter."""
    if len(seq) <= 2:
        return seq
    letters = set(seq)
    last_char = seq[-1]
    successors = {c: [] for c in letters}
    for a, b in zip(seq, seq[1:]):
        successors[a].append(b)

    # 1. choose a random "last exit" edge for every letter except the final one,
    #    retrying until those edges form a tree pointing at the final letter
    while True:
        last_edges = {x: rng.choice(successors[x]) for x in letters if x != last_char and successors[x]}
        if _connected_to_last(last_edges, last_char, {x for x in letters if successors[x]} | {last_char}):
            break

    # 2. shuffle the remaining edges of each letter, then append its last-exit edge
    lists = {}
    for x in letters:
        rest = list(successors[x])
        if x in last_edges:
            rest.remove(last_edges[x])
        rng.shuffle(rest)
        if x in last_edges:
            rest.append(last_edges[x])
        lists[x] = rest

    # 3. walk the Eulerian path
    out = [seq[0]]
    current = seq[0]
    for _ in range(len(seq) - 1):
        nxt = lists[current].pop(0)
        out.append(nxt)
        current = nxt
    return "".join(out)


def dinucleotide_counts(seq: str) -> Counter:
    return Counter(seq[i:i + 2] for i in range(len(seq) - 1))


# ============================================================
# READING DATASETS
# ============================================================

def _clean(seq: str) -> str:
    return "".join(seq.split()).upper()


def read_sequences(path: str | Path, seq_len: int | None = 101, seed: int = 42,
                   make_negatives_for_fasta: bool = True) -> tuple[list[str], list[str], np.ndarray]:
    """Return (ids, sequences, labels) and drop rows with wrong length or non-ACGT letters."""
    path = Path(path)
    text = path.read_text(encoding="utf-8", errors="replace")
    first = next((ln for ln in text.splitlines() if ln.strip()), "")

    ids, seqs, labels = [], [], []
    if first.startswith(">") and len(first.split()) == 1:
        # FASTA: header line, then sequence line(s)
        cur_id, buf = None, []
        for ln in text.splitlines() + [">__end__"]:
            if ln.startswith(">"):
                if cur_id is not None:
                    ids.append(cur_id); seqs.append(_clean("".join(buf))); labels.append(1)
                cur_id, buf = ln[1:].strip(), []
            else:
                buf.append(ln)
        if make_negatives_for_fasta:
            rng = random.Random(seed)
            for i, s in list(zip(ids, seqs)):
                ids.append(i + "_shuf"); seqs.append(dinucleotide_shuffle(s, rng)); labels.append(0)
    elif "," in first and "sequence" in first.lower():
        with open(path, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                row = {k.strip().lower(): v for k, v in row.items()}
                ids.append(row.get("chr_id") or row.get("id") or "")
                seqs.append(_clean(row["sequence"]))
                labels.append(int(float(row["label"])))
    else:
        # ENCODE benchmark rows: id sequence label (whitespace separated)
        for ln in text.splitlines():
            parts = ln.split()
            if len(parts) < 3:
                continue
            ids.append(parts[0].lstrip(">"))
            seqs.append(_clean(parts[1]))
            labels.append(int(float(parts[2])))

    keep_ids, keep_seqs, keep_labels, seen = [], [], [], set()
    for i, s, y in zip(ids, seqs, labels):
        if seq_len and len(s) != seq_len:
            continue
        if not s or set(s) - VALID or y not in (0, 1) or (s, y) in seen:
            continue
        seen.add((s, y))
        keep_ids.append(i); keep_seqs.append(s); keep_labels.append(y)
    return keep_ids, keep_seqs, np.array(keep_labels, dtype=np.int64)


def write_benchmark(path: str | Path, ids, seqs, labels) -> None:
    """Write in the same row format as the ENCODE benchmark (works with read_sequences)."""
    with open(path, "w", encoding="utf-8") as fh:
        for i, s, y in zip(ids, seqs, labels):
            fh.write(f">{i} {s} {int(y)}\n")


def one_hot_batch(seqs: list[str]) -> np.ndarray:
    """(N, 4, L) float32 one-hot in A, C, G, T order."""
    lut = np.full(256, -1, dtype=np.int64)
    for k, c in enumerate("ACGT"):
        lut[ord(c)] = k
    arr = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(len(seqs), -1)
    idx = lut[arr]
    out = np.zeros((len(seqs), 4, arr.shape[1]), dtype=np.float32)
    n, l = np.nonzero(idx >= 0)
    out[n, idx[n, l], l] = 1.0
    return out


# ============================================================
# MINIMAL .2bit GENOME READER
# ============================================================

class TwoBitReader:
    """Random access to a UCSC .2bit file (format: https://genome.ucsc.edu/FAQ/FAQformat.html#format7)."""

    _BASES = "TCAG"

    def __init__(self, path: str | Path):
        self.fh = open(path, "rb")
        sig = self.fh.read(4)
        if sig == struct.pack("<I", 0x1A412743):
            self.end = "<"
        elif sig == struct.pack(">I", 0x1A412743):
            self.end = ">"
        else:
            raise ValueError("Not a .2bit file")
        version, count, _ = struct.unpack(self.end + "III", self.fh.read(12))
        if version != 0:
            raise ValueError("Only 2bit version 0 is supported")
        self.index = {}
        for _ in range(count):
            n = self.fh.read(1)[0]
            name = self.fh.read(n).decode()
            (offset,) = struct.unpack(self.end + "I", self.fh.read(4))
            self.index[name] = offset
        self._meta = {}

    def _record(self, chrom):
        if chrom not in self._meta:
            self.fh.seek(self.index[chrom])
            (size,) = struct.unpack(self.end + "I", self.fh.read(4))
            (nb,) = struct.unpack(self.end + "I", self.fh.read(4))
            n_starts = struct.unpack(self.end + f"{nb}I", self.fh.read(4 * nb))
            n_sizes = struct.unpack(self.end + f"{nb}I", self.fh.read(4 * nb))
            (mb,) = struct.unpack(self.end + "I", self.fh.read(4))
            self.fh.seek(8 * mb + 4, 1)  # skip mask blocks + reserved
            self._meta[chrom] = (size, list(zip(n_starts, n_sizes)), self.fh.tell())
        return self._meta[chrom]

    def chrom_size(self, chrom):
        return self._record(chrom)[0]

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """0-based, half-open [start, end). Returns uppercase sequence with N blocks."""
        size, nblocks, dna_off = self._record(chrom)
        if start < 0 or end > size or start >= end:
            raise ValueError(f"{chrom}:{start}-{end} outside 0-{size}")
        first, last = start // 4, (end - 1) // 4
        self.fh.seek(dna_off + first)
        raw = self.fh.read(last - first + 1)
        chars = []
        for byte in raw:
            for shift in (6, 4, 2, 0):
                chars.append(self._BASES[(byte >> shift) & 3])
        s = list("".join(chars)[start - first * 4: start - first * 4 + (end - start)])
        for ns, nl in nblocks:
            a, b = max(ns, start), min(ns + nl, end)
            for p in range(a, b):
                s[p - start] = "N"
        return "".join(s)
