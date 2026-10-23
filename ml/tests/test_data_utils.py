"""Tests for data loading, dinucleotide shuffling, .2bit reading and dataset building.
Run:  cd ml && python -m pytest tests -q   (or: python tests/test_data_utils.py)"""
import gzip, random, struct, sys, tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_utils import (dinucleotide_shuffle, dinucleotide_counts, read_sequences, write_benchmark,
                        one_hot_batch, TwoBitReader)


def test_shuffle_keeps_dinucleotides():
    rng = random.Random(0)
    for _ in range(300):
        n = rng.randint(3, 120)
        s = "".join(rng.choice("ACGT") for _ in range(n))
        t = dinucleotide_shuffle(s, rng)
        assert len(t) == len(s) and t[0] == s[0] and t[-1] == s[-1]
        assert dinucleotide_counts(t) == dinucleotide_counts(s)
    # it really shuffles
    s = "ACGTTGCAAGCTTCGA" * 6
    assert len({dinucleotide_shuffle(s, rng) for _ in range(20)}) > 10


def test_read_formats(tmp=None):
    d = Path(tempfile.mkdtemp())
    seq = "A" * 50 + "C" * 51
    (d / "a.data").write_text(f">chr1:0-101 {seq} 1\n>chr1:0-101_shuf {seq[::-1]} 0\n>bad ACGN 1\n")
    (d / "b.csv").write_text(f"chr_id,sequence,label\nchr1,{seq.lower()},1\nchr2,{'G'*101},0\n")
    (d / "c.fa").write_text(f">p1\n{seq[:60]}\n{seq[60:]}\n")
    ids, seqs, y = read_sequences(d / "a.data")
    assert seqs == [seq, seq[::-1]] and list(y) == [1, 0]
    ids, seqs, y = read_sequences(d / "b.csv")
    assert seqs[0] == seq and list(y) == [1, 0]
    ids, seqs, y = read_sequences(d / "c.fa")
    assert list(y) == [1, 0] and ids[1] == "p1_shuf" and dinucleotide_counts(seqs[1]) == dinucleotide_counts(seq)
    write_benchmark(d / "w.data", ids, seqs, y)
    assert read_sequences(d / "w.data")[1] == seqs


def test_one_hot():
    x = one_hot_batch(["ACGT", "TTTT"])
    assert x.shape == (2, 4, 4) and x[0, :, 0].tolist() == [1, 0, 0, 0] and x[1, 3].sum() == 4


def _write_2bit(path, seqs):
    code = {"T": 0, "C": 1, "A": 2, "G": 3, "N": 0}
    header = struct.pack("<IIII", 0x1A412743, 0, len(seqs), 0)
    index_size = sum(1 + len(n) + 4 for n in seqs)
    offset = len(header) + index_size
    index, body = b"", b""
    for name, s in seqs.items():
        index += bytes([len(name)]) + name.encode() + struct.pack("<I", offset)
        nblocks, i = [], 0
        while i < len(s):
            if s[i] == "N":
                j = i
                while j < len(s) and s[j] == "N":
                    j += 1
                nblocks.append((i, j - i)); i = j
            else:
                i += 1
        rec = struct.pack("<II", len(s), len(nblocks))
        rec += struct.pack(f"<{len(nblocks)}I", *[a for a, _ in nblocks])
        rec += struct.pack(f"<{len(nblocks)}I", *[b for _, b in nblocks])
        rec += struct.pack("<II", 0, 0)
        padded = s + "T" * (-len(s) % 4)
        rec += bytes(sum(code[padded[k + j]] << (6 - 2 * j) for j in range(4)) for k in range(0, len(padded), 4))
        body += rec
        offset += len(rec)
    path.write_bytes(header + index + body)


def test_twobit_and_builder():
    import build_encode_dataset as b
    rng = random.Random(1)
    d = Path(tempfile.mkdtemp())
    chr1 = "".join(rng.choice("ACGT") for _ in range(5003)) + "NNNNN" + "".join(rng.choice("ACGT") for _ in range(997))
    chr2 = "".join(rng.choice("ACGT") for _ in range(2000))
    _write_2bit(d / "g.2bit", {"chr1": chr1, "chr2": chr2})
    g = TwoBitReader(d / "g.2bit")
    assert g.chrom_size("chr1") == len(chr1)
    for s, e in [(0, 101), (3, 7), (4990, 5020), (5900, 6005)]:
        assert g.fetch("chr1", s, e) == chr1[s:e]
    # fake narrowPeak: 3 good peaks, 1 overlapping Ns, 1 too close to the edge
    rows = [("chr1", 1000, 1400, 9.0, 200), ("chr1", 2000, 2300, 8.0, -1), ("chr2", 500, 900, 7.0, 10),
            ("chr1", 4950, 5100, 6.0, 60), ("chr2", 0, 60, 5.0, 5)]
    cache = d / "cache"; cache.mkdir()
    with gzip.open(cache / "TEST.narrowPeak.gz", "wt") as fh:
        for c, s, e, sig, summ in rows:
            fh.write(f"{c}\t{s}\t{e}\t.\t0\t.\t{sig}\t-1\t-1\t{summ}\n")
    stats = b.build_one("TEST", g, d / "out", max_peaks=100, test_frac=0.34, seed=0, cache=cache)
    assert stats["train"] + stats["test"] == 6  # 3 positives + 3 shuffled negatives
    ids, seqs, y = read_sequences(d / "out" / "TEST" / "train.data")
    for i, s, lab in zip(ids, seqs, y):
        if lab == 1:
            chrom, rng_ = i.split(":"); a, e = map(int, rng_.split("-"))
            assert s == {"chr1": chr1, "chr2": chr2}[chrom][a:e] and len(s) == 101
    assert "chr1:1150-1251" in ids + read_sequences(d / "out" / "TEST" / "test.data")[0]  # summit 1200 ±50


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
