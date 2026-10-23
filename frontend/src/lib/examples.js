// Real 101 bp sequences from the FOSL2 (HepG2) ENCODE test set the model is evaluated on.
// label 1 = FOSL2 bound this region in the ChIP-seq experiment; label 0 = dinucleotide-shuffled control.
export const REAL_EXAMPLES = [
  {
    "id": "chr8:39917210-39917310",
    "label": 1,
    "kind": "bound (has AP-1 site)",
    "sequence": "TCCTGCGTCTAAAGGCACTTGTCTACCAAACACTTCCTTTTAGAAAGTACTAATCAGTGACTCAGCCTGGTGACTCCTCACCCTAAAGTCCCTTTGGCTTC"
  },
  {
    "id": "chr1:31944499-31944599",
    "label": 1,
    "kind": "bound (has AP-1 site)",
    "sequence": "GTGGTTACAGAACAGTTGACTCAGGATGATTCAGGTCAGAGCAGGTGACAGGGGTGACTCAAGATGGAGCAGGTGACCAGGGGTGACTCAGGACAAAGCAG"
  },
  {
    "id": "chr10:100220683-100220783",
    "label": 1,
    "kind": "bound (has AP-1 site)",
    "sequence": "TCAGCAGAAAAGGCCAGAAACAGGCCACCAGCTGGAAATAAGGTTGTGAGTCACCCTCAGAAAATCAACATTGCATAGTGATGACTGTTTCTGGGCCTTGC"
  },
  {
    "id": "chr2:163436612-163436712",
    "label": 1,
    "kind": "bound (no obvious AP-1 site)",
    "sequence": "TATGACCTTTTTGAGCTTGTGCTAAAGGCATCATAGCTCACAGTCAGGAAGGTCTTTCTAACTGAACCCTTGTCCTCTGGCAGTGCTAACTTGTTTACATT"
  },
  {
    "id": "chr20:307938-308038_shuf",
    "label": 0,
    "kind": "not bound (shuffled control)",
    "sequence": "GGTTACGGGCCCGAAGGCACCGCGGGCCCCCCCCCCCTCCCAGGCGGGATCACTCTCCTGCGCCAGGCCGGCTTGGGCCCTCTGCTGGGCCGCGCCCTTTA"
  },
  {
    "id": "chr9:104077470-104077570_shuf",
    "label": 0,
    "kind": "not bound (shuffled control)",
    "sequence": "CTACATGAGTTAAGAGAATGAAGTCCTTCAAAACCCTGCCTCGGGATCTTACTTTATTGACATTTTTCCTTTTTACCCTCTATCAATTACTTTCCCCTTGA"
  }
];
