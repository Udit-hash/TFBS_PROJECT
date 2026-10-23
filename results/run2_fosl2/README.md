# Run 2: FOSL2 (HepG2), final recipe

Same data, folds and test set as run 1. Every deep model used the same recipe: early stopping on
validation AUROC (patience 6, max 30 epochs), reverse-complement augmentation, AdamW (lr 3e-4).
Added: tri-branch v2 (deeper shared convolution trunk) and its two ablations.

Best by mean validation AUROC (never by test): CNN + BiLSTM (validation 0.941; test 0.939 ± 0.002).
The website's default is the team's proposed tri-branch v2 (test 0.934 ± 0.002), with a switch to
CNN + BiLSTM, CNN and v1. Removing the Transformer
(0.934) or the BiLSTM (0.933) from v2 changes nothing, so on this dataset the extra branches add no
accuracy once the convolution trunk is deep enough.

`interpretation/` was computed on the selected CNN + BiLSTM model (7 bp first-layer filters, so its
logos are shorter and weaker than run 1's tri-branch filters in ../run1_fosl2/interpretation).
