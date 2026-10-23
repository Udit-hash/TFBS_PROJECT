# Run 1: FOSL2 (HepG2), first training recipe

Trained in Google Colab (T4 GPU) with `ml/train_all.py` as of commit 442d65e:
5-fold cross-validation, early stopping on validation F1 at a 0.5 cut-off, no strand augmentation.
Test set: 9,920 held-out sequences, never used for training or for choosing the cut-off.

Main finding: CNN + BiLSTM was the best model (AUROC 0.928 ± 0.003). The original tri-branch model
(one thin convolution layer, 24 filters per size) scored 0.897 ± 0.009, below both CNN baselines.
This led to run 2 (deeper tri-branch "v2", the same improved recipe for all deep models, and an ablation).

Files: `results_table.md` (main table), `roc_curves.png`, `cv_results.json` (every fold),
`interpretation/` (filter logos matched to JASPAR and the explanation-method comparison, computed on the
run-1 tri-branch model).
