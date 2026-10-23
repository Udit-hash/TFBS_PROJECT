# Changes needed in the research paper

This compares the paper (`TFBS_ResearchPaper_1.docx`) with what the project actually does and measured. Every number below comes from `results/run2_fosl2/` (final run) or `results/run1_fosl2/` (first run), on 9,920 held-out test sequences that were never used for training or tuning.

## 1. Main results (put in a new Section VI "Results", replacing "Expected Outcomes")

**TABLE II. FOSL2 (HepG2) test performance, mean ± SD over 5 cross-validation folds**

| Method | AUROC | PR-AUC | Accuracy | F1 | MCC |
|---|---|---|---|---|---|
| JASPAR PWM scan (MA0478.2) | 0.785 | 0.834 | 0.752 ± 0.015 | 0.717 ± 0.002 | 0.518 ± 0.042 |
| k-mer (k = 5) logistic regression | 0.893 | 0.898 | 0.808 ± 0.003 | 0.811 ± 0.001 | 0.617 ± 0.005 |
| CNN | 0.924 ± 0.003 | 0.932 ± 0.002 | 0.845 ± 0.003 | 0.844 ± 0.002 | 0.690 ± 0.005 |
| CNN + BiLSTM | **0.939 ± 0.002** | **0.944 ± 0.001** | **0.864 ± 0.004** | **0.863 ± 0.003** | **0.729 ± 0.007** |
| CNN + BiLSTM + Transformer, v1 | 0.911 ± 0.008 | 0.919 ± 0.007 | 0.825 ± 0.010 | 0.825 ± 0.010 | 0.650 ± 0.019 |
| **CNN + BiLSTM + Transformer, v2 (proposed)** | 0.934 ± 0.002 | 0.940 ± 0.002 | 0.855 ± 0.004 | 0.854 ± 0.002 | 0.710 ± 0.008 |

(The PWM and k-mer rows have ± 0.000 for AUROC because they barely change between folds; write them without ±.)

**TABLE III. Ablation of the proposed model (v2)**

| Variant | AUROC | MCC |
|---|---|---|
| Full (CNN + BiLSTM + Transformer) | 0.934 ± 0.002 | 0.710 ± 0.008 |
| without Transformer | 0.934 ± 0.001 | 0.715 ± 0.002 |
| without BiLSTM | 0.933 ± 0.000 | 0.710 ± 0.006 |

**Suggested text (Results):**
> On the FOSL2 HepG2 benchmark, all deep models clearly outperformed the classic baselines (JASPAR PWM AUROC 0.785; k-mer logistic regression 0.893). The proposed tri-branch model reached an AUROC of 0.934 ± 0.002, 0.010 above the CNN-only baseline and 0.141 above the PWM baseline. The best model was CNN + BiLSTM (0.939 ± 0.002). The first version of the tri-branch model, which used a single convolution layer with 24 filters per kernel size, reached only 0.911 ± 0.008. Adding a second shared convolution layer (v2) closed most of the gap, which shows that the depth of the motif-detection stage matters more than the choice of sequence-context branch. The ablation (Table III) confirms this: removing the Transformer or the BiLSTM branch from v2 left the AUROC unchanged (0.934 and 0.933). For 101 bp windows centred on ChIP-seq peaks, local motif features dominate, and the long-range context that self-attention is designed to capture adds no measurable accuracy.

Figures to include: `results/run2_fosl2/roc_curves.png` (Fig. ROC), `results/run1_fosl2/interpretation/filter_logos.png` (Fig. learned motifs), `results/run2_fosl2/interpretation/attribution_example.png` (Fig. attribution example).

## 2. Interpretability results (add to Results)

**TABLE IV. How well each explanation method finds the real FOSL2 site** (500 bound test sequences with a strong JASPAR MA0478.2 match; selected model)

| Method | Core AUROC | Top position on the site | Importance on the site |
|---|---|---|---|
| ShiftSmooth (proposed) | **0.842 ± 0.219** | 69.4% | 22.3% |
| Gradient (saliency) | 0.821 ± 0.219 | 67.8% | 22.3% |
| Gradient × input | 0.759 ± 0.194 | **69.8%** | **27.0%** |
| Integrated Gradients, shuffled references | 0.722 ± 0.185 | 58.6% | 17.5% |
| Integrated Gradients, flat reference | 0.714 ± 0.189 | 56.2% | 21.3% |
| Random (chance) | 0.499 | 14.0% | 9.9% |

> ShiftSmooth gave the best separation between motif and non-motif positions (core AUROC 0.842 vs 0.821 for plain gradients and 0.72 for Integrated Gradients), supporting its use in the system. All gradient-based methods were far above chance. Integrated Gradients performed worst, which is consistent with known difficulties of path-based attribution in max-pooled convolutional networks.

**Learned motifs (run 1, tri-branch v1 filters, 71 informative filters):**
> Seven first-layer filters matched AP-1 family motifs in JASPAR, including FOS (MA0476.2, Pearson r = 0.98–0.99), FOSL2 itself (MA0478.2, r = 0.97) and FOS::JUNB (r = 0.95), recovering the canonical TGA(C/G)TCA site. Other filters matched liver-specific factors: FOXA2 (r = 0.93), HNF4A (r = 0.85), FOXA3 and NR1H4/FXR. These are known co-factors at HepG2 enhancers, so the model learned both the FOSL2 site and the cell-type context it binds in.

Motif similarity is the mean per-column Pearson correlation at the best alignment (both strands), computed against all 1,019 JASPAR 2024 CORE vertebrate matrices. **For the E-values the paper promises:** upload `results/run1_fosl2/interpretation/filters.meme` to Tomtom (https://meme-suite.org/meme/tools/tomtom) against "JASPAR CORE (2024) vertebrates" and copy the E-values for the AP-1 filters. Otherwise, delete the E-value sentences.

## 3. Section-by-section edits

**Abstract**
- "trained and evaluated on 165 ENCODE ChIP-seq benchmark datasets" → "trained and evaluated on the ENCODE FOSL2 HepG2 ChIP-seq benchmark (one of the 165 datasets used by DeepSTF and MLSNet), using five-fold cross-validation".
- "is expected to demonstrate superior performance …" → state the result: tri-branch AUROC 0.934, above CNN (0.924), k-mer (0.893) and PWM (0.785) baselines and competitive with CNN + BiLSTM (0.939).
- Remove "sequence logo generation … correspondence to curated JASPAR entries" claims that aren't backed by results, or keep them now that they are (learned filters matched FOS/FOSL2 at r ≥ 0.97).
- Remove "DNA shape" mentions.

**I. Introduction, contribution (c):** "demonstrating state-of-the-art performance across diverse transcription factors" → "a controlled comparison against classic and deep baselines on a real ENCODE dataset, with an ablation of each branch and a quantitative comparison of attribution methods".

**III. Problem statement:** delete the objective "to integrate sequence shape and biophysical features as complementary modalities" (move to Future Work).

**IV-A. Data:**
- "fixed length (200 bp)" → **101 bp**. "4 × 200" → **4 × 101**.
- Delete the DNA-shape paragraph ("A secondary encoding branch computes five DNA shape features …").
- Add: 39,439 training and 9,920 test sequences, half bound; negatives are dinucleotide-shuffled positives; source: Zeng et al. 2016 benchmark (ref [2]) via the BERT-TFBS repository.

**IV-B. Models:** describe what was actually trained:
- v2: parallel convolutions of width 8, 16 and 24 bp (32 filters each) → batch norm, ReLU, max-pool → 5 bp convolution (128 channels) → max-pool → three branches: global max-pool (CNN), BiLSTM (64 units per direction) and a one-layer Transformer encoder (4 heads, feed-forward size 256, sinusoidal positions). Concatenation → 128-unit GELU layer → sigmoid output.
- Training: AdamW (learning rate 3 × 10⁻⁴, weight decay 5 × 10⁻⁴), batch 64, up to 30 epochs, learning rate halved on plateau, early stopping on validation AUROC (patience 6), reverse-complement augmentation. The decision cut-off maximises F1 on the validation fold.
- Equations (1) and (2) are fine.

**IV-C. Visualisation:**
- Logos: "filter activations above 50% of each filter's maximum over bound test sequences are aligned into position frequency matrices"; similarity to JASPAR by column-wise Pearson correlation (and Tomtom if you run it).
- Remove FIMO and MEME unless you run them. The project scans motifs with its own log-odds PWM scanner on both strands.
- ShiftSmooth: averages |gradient| over cyclic shifts of −2…+2 bp.
- Add: Monte-Carlo dropout (30 passes) gives a 95% interval for each prediction. This is the "confidence interval" in Section V.

**IV-D. Evaluation:**
- "Experiments are conducted on 165 ENCODE ChIP-seq benchmark datasets" → one dataset (FOSL2 HepG2). The script `ml/build_encode_dataset.py` can rebuild the others for future work.
- Metrics: add PR-AUC and MCC.
- "implemented in Python using TensorFlow and PyTorch, with OpenCV and Matplotlib" → "PyTorch, scikit-learn, NumPy and Matplotlib". TensorFlow and OpenCV aren't used.

**V. System architecture:**
- "independent machine learning and visualization microservices" → "a FastAPI backend with separate modules for prediction, explanation and motif analysis".
- Table I: keep the layers, replacing "PyTorch/TF" with "PyTorch".
- Add the new components: the 3D double-helix view coloured by attribution, the PDB structure viewer (e.g. 1FOS, the c-Fos/c-Jun AP-1 complex), the JASPAR scan of the input, the model switch, and the beginner/advanced tutor chatbot.

**VI. Expected Outcomes** → rename "Results and Discussion" and replace with sections 1–2 above. Remove the unsupported claims: "AUROC exceeding 0.92 across the majority of the 165 datasets", "competitive with Enformer and Sei", "4–8 percentage point gains", "2–5% AUROC from shape features".

**VII. Conclusion, future work:** add (1) more TFs and cell lines (all 165 datasets), (2) genomic-background negatives instead of shuffles, (3) DNA shape features (planned, not implemented), (4) longer input windows where self-attention may help, (5) the existing items (cooperative binding, epigenomic context, pre-training).

**References:** add JASPAR 2024 (Rauluseviciute et al., Nucleic Acids Res. 52:D174, 2024). [15] is JASPAR 2016; update it or add 2024. Add the BERT-TFBS repository as the data source if your format allows. Remove references only cited for things not done (e.g. [12] Walters & Barzilay is fine as general context).

## 4. Honest framing for the viva

- **"Why isn't your proposed model the best?"** It is within 0.005 AUROC of the best, clearly beats CNN-only and classic methods, and the ablation shows *why*: on 101 bp windows the extra branches don't add information. Testing a design and reporting that honestly is a valid scientific result.
- **"Did you tune on the test set?"** No. Every choice (cut-off, early stopping, which model is "best") used validation data only. The test set was scored once per fold.
- **Run 1 vs run 2:** run 1 found the tri-branch v1 was too shallow. Run 2 fixed the architecture and used one training recipe for **every** deep model, so the comparison stayed fair. Both runs are in `results/`.
