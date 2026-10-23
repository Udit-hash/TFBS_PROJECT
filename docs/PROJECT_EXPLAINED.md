# The TFBS project, explained in plain language

This explains every part of the project so anyone on the team can explain it in the viva.

## 1. The question the project answers

A **transcription factor** (TF) is a protein that switches genes on or off by sticking to short "words" in DNA. The place it sticks is its **binding site** (TFBS).

Our project asks: *given a 101-letter piece of human DNA, will the TF **FOSL2** stick to it?* It then explains **why** the model thinks so.

FOSL2 belongs to the **AP-1** family (with Fos and Jun). Its favourite word is **TGACTCA** (or TGAGTCA).

## 2. The data

- **Source:** ENCODE ChIP-seq experiment for FOSL2 in HepG2 (liver) cells: `wgEncodeAwgTfbsHaibHepg2Fosl2V0416101UniPk`. It is one of the 165 ENCODE datasets used by DeepSTF and MLSNet, and part of the 690-dataset benchmark from Zeng et al. 2016. We download it from the public BERT-TFBS repository.
- **Positives (label 1):** 101 bp windows centred on places where FOSL2 was measured sitting on the DNA.
- **Negatives (label 0):** the same sequences **dinucleotide-shuffled**. The letters are rearranged but every pair of neighbouring letters (AA, AC, …) appears the same number of times. The composition stays identical but the binding word is destroyed, so the model has to learn the *word*, not just "lots of G and C".
- **Size:** about 39,400 training and 9,900 test sequences, half bound and half not.
- `ml/make_synthetic_sequences.py` makes **fake** random sequences. It exists only for the mentor's "random input" sanity check and is never used for training.

## 3. How DNA goes into the model

Each letter becomes four switches: A = 1000, C = 0100, G = 0010, T = 0001 (**one-hot encoding**). A sequence becomes a 4 × 101 grid of 0s and 1s.

## 4. The models we compare (`ml/train_all.py`)

| Method | What it is | Why it's in the table |
|---|---|---|
| JASPAR PWM scan | Slides FOSL2's known motif from the JASPAR database along the sequence and takes the best match. No learning. | The "old school" bioinformatics baseline the paper says we beat |
| k-mer logistic regression | Counts every 5-letter word, then a simple linear classifier | The classic machine-learning baseline (like the SVM/k-mer methods in Related Work) |
| CNN | Convolution filters slide along the DNA looking for short patterns | Shows what convolution alone achieves |
| CNN + BiLSTM | CNN patterns, then a recurrent network reads them left→right and right→left | Shows what adding sequence context gives |
| **CNN + BiLSTM + Transformer (ours)** | Three CNN filter sizes (8, 16, 24 bp) feed both a BiLSTM and a Transformer (self-attention). All three summaries are joined and a small classifier outputs a probability | The paper's model |

## 5. How training and testing work

- **The test set is never used for learning**, only for the final scores.
- **5-fold cross-validation:** the training set is cut into 5 parts. Each model is trained 5 times; each time 4 parts teach the model and the 5th part (validation) tells it when to stop (early stopping) and picks the cut-off. All 5 versions are then scored on the same test set, and we report **mean ± standard deviation**. That shows the result isn't a lucky split.
- **Cut-off:** a probability becomes "bound / not bound" by comparing it to a cut-off. Instead of always using 0.5, we choose the cut-off that gives the best F1 on the validation part. The chosen cut-off is saved inside the model file, and the website uses it.
- **Metrics:** AUROC (main number: how well bound sequences are ranked above unbound ones; 0.5 = guessing, 1 = perfect), PR-AUC, accuracy, precision, recall, F1 and MCC (−1…1, 0 = guessing).

## 6. Explaining the model (`ml/xai.py`, `ml/interpret.py`, website)

- **Saliency (gradient):** "if this letter changed a tiny bit, how much would the score change?" Big = important.
- **ShiftSmooth:** saliency computed on the sequence nudged 2 letters left/right and averaged, which removes noise. This is the method the paper proposes.
- **Gradient × input** and **Integrated Gradients:** two standard alternatives we compare against.
- **In-silico mutagenesis (ISM):** actually change each letter to each other letter (303 mutants) and re-predict. If the score drops a lot, that letter matters.
- **Which explanation method is best?** In bound test sequences we locate the real FOSL2 site with the JASPAR motif. Then we check how often each method's most important position lands on that site ("peak on motif") and how well its importance separates site from non-site positions ("core AUROC").
- **Filter logos:** each first-layer CNN filter is a little pattern detector. We collect the DNA pieces that switch each filter on, stack them into a **sequence logo**, and compare that logo with all ~900 JASPAR motifs. If training worked, some filters look like the AP-1 site TGA(C/G)TCA. `filters.meme` can also be uploaded to the official **Tomtom** tool (meme-suite.org) to get the E-values the paper mentions.
- **Confidence interval:** the website runs the model 30 extra times with dropout switched on. The spread of those answers gives a 95% interval (**Monte-Carlo dropout**).
- **JASPAR scan on the website:** your sequence is scanned with every JASPAR motif. The card shows the known motif that sits where the model is looking.

## 7. The website

- `frontend/` (React): paste a sequence, upload a FASTA file, load a real FOSL2 example, or make random DNA with an AP-1 site planted in it.
- `ml/api.py` (FastAPI): `/predict` returns the probability, interval, cut-off, saliency, ISM and JASPAR hits; `/chat` answers questions.
- **3D DNA helix:** your sequence drawn as a double helix, base pairs coloured by importance, with hydrogen bonds (A–T has 2, G–C has 3). It's a drawing of the model's scores on idealised DNA, not a physics simulation.
- **Real structure tab:** a lab-measured crystal structure from the Protein Data Bank. For FOSL2 it shows c-Fos/c-Jun bound to an AP-1 site (PDB 1FOS).
- **DNA tutor chatbot:** answers beginner or advanced questions and knows your current result. It works offline from `ml/chatbot/knowledge_base.json`, or uses a free online AI model if a key is added in `ml/.env`.

## 8. Contributions from Apoorva's commit (Oct 21) that are still in the project

- **FASTA upload** on the website (`App.jsx`): reads a `.fa/.fasta` file with one record and checks it only has A/C/G/T.
- **New page design** (`App.css`) and the **`/api` proxy** in `frontend/vite.config.js`: the website calls `/api/...` and Vite forwards it to the backend on port 8001, so the browser doesn't hit cross-origin problems.
- **CNN + BiLSTM model** (`ml/models/cnn_bilstm.py`): two convolution layers (7 and 5 bp) then a BiLSTM. It's the middle step between the CNN and our full model in the comparison.
- **Bigger tri-branch model** (`ml/models/hybrid.py`): 24 filters per size instead of 16, a 48-unit BiLSTM, GELU activations and a two-layer classifier, plus `encode_sequence()`, which returns the joined features before the classifier.
- **Training settings** (`ml/config.py`, `ml/train.py`): learning rate 3e-4, AdamW with weight decay 5e-4, learning-rate halving when validation F1 stops improving, and saving the best model by validation F1.
- **Loading old model files** (now in `ml/models/loader.py`): reads the layer sizes from a saved file, so older checkpoints with different sizes still load.
- **Removed:** her DNA-shape feature, because it needed the R package DNAshapeR. The project is Python + JavaScript only, and DNA shape is listed as future work.

## 9. Likely viva questions

- **Why shuffled negatives?** They keep the letter composition identical, so the model must learn the binding word itself. *Limitation:* real genomic background is harder; mention it as future work.
- **Why 5-fold CV and a separate test set?** CV shows stability (± SD); the untouched test set gives an honest final score.
- **Why tune the cut-off on validation, not test?** Choosing anything using the test set would leak information and inflate the scores.
- **Does the 3D view show real bonds?** It shows real base-pairing chemistry (2 or 3 hydrogen bonds) on idealised DNA, coloured by the model's attributions. The model itself only reads letters. The real-structure tab shows actual protein–DNA contacts from crystallography.
- **How do you know the model learned biology?** The filter logos match JASPAR's AP-1/FOSL2 motif, and the attribution benchmark shows the important positions land on real FOSL2 sites.
