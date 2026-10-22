# TFBS Prediction Project

## Existing project audit

- **Input length:** the active configuration is 101 bp (`ml/config.py`), and the
  preprocessing, data preparation, API, and React form all enforce 101 bp.
  The transformer's positional-encoding capacity of 200 is only a maximum; it
  does not make the current dataset or model a 200-bp model.
- **Sequence encoding:** canonical A/C/G/T bases become one-hot arrays of shape
  `[4, 101]`; model batches use `[batch, 4, 101]`.
- **Models found in source:** `TFBS_CNN` and `TFBS_TriBranch` (multi-scale CNN,
  BiLSTM, and Transformer). A standalone CNN+BiLSTM model was not present in
  the inspected source; `ml/models/cnn_bilstm.py` adds it as a separate model
  for the requested comparison. `best_cnn.pth` and `best_hybrid.pth` are the
  existing checkpoints. The current hybrid implementation does not load the
  existing hybrid checkpoint, and `ml/api.py` falls back to the CNN. The
  existing checkpoints are not overwritten by the comparison run.
- **Dataset contract:** raw and processed CSVs use `chr_id`, `sequence`, and
  binary `label` columns. The existing preparation keeps the separate train
  and test CSVs; training selects validation data from the training CSV and
  leaves the test CSV for final evaluation. No CSV dataset files are present
  in this checkout. `ml/fetch_encode_data.py` generates synthetic motif-based
  sequences; despite its comments, those examples are not real ENCODE data.
- **Existing inference:** `ml/api.py` loads a sequence model and returns a
  binding probability, classification, motif list, and visualizations.
  `frontend/src/App.jsx` calls that API through the Vite `/api` proxy.
  `backend/main.py` is a separate heuristic API and is not the model API used
  by the frontend's configured proxy.
- **Existing training utilities:** `ml/train.py` trains only the tri-branch
  model. `ml/evaluate.py` and `ml/predict.py` are empty. The saved
  `hybrid_results.json` labels its model `Hybrid_CNN_BiLSTM`, which does not
  match the current tri-branch class name or its incompatible checkpoint.
  Those saved metrics should not be treated as results for the new comparison.

The DNA-shape fusion model has new trainable shape and fusion layers, and its
sequence branch must be fitted together with them. The old sequence-only
checkpoints are not DNA-shape-enhanced and cannot be promoted by loading them
unchanged. The comparison script trains all four configurations from scratch
on the same split and writes separate checkpoints; it does not overwrite the
existing CNN or hybrid checkpoints.

The 101-bp setting is deliberately unchanged. If the research protocol needs
200-bp windows, update sequence generation, data preparation, configuration,
all model/training/inference validation, and the frontend together, prepare
new 200-bp train/validation/test splits, then retrain every model. Do not mix
101-bp checkpoints or features with a 200-bp evaluation.

## DNA-shape prediction

The new branch uses **DNAshapeR**, the R/Bioconductor implementation of the
DNAshape method. It uses a sliding pentamer lookup table whose structural
values are based on all-atom Monte Carlo simulations. It predicts MGW and
ProT (base-pair parameters) and Roll and HelT (base-pair-step parameters);
these are numerical predictions, not direct physical measurements. The
conventional units are MGW in angstroms and the angular parameters in degrees.
DNAshapeR documents FASTA input and exposes these outputs through `getShape`.
Only A/C/G/T is accepted here; ambiguous symbols are rejected rather than
imputed.

The reviewed Bioconductor release documentation identifies DNAshapeR 1.40.0
in Bioconductor 3.23. `getShape(fasta, shapeType = "All")` returns the
prediction object containing feature-specific results; the bridge exports
those arrays to tab-separated numeric matrices for Python. Install a supported
R release and let `BiocManager` select the compatible Bioconductor build.

References:

- [DNAshapeR on Bioconductor](https://bioconductor.org/packages/release/bioc/html/DNAshapeR.html)
- [DNAshapeR vignette](https://bioconductor.org/packages/release/bioc/vignettes/DNAshapeR/inst/doc/DNAshapeR.html)
- Zhou et al., [DNAshape: a method for the high-throughput prediction of DNA structural features](https://doi.org/10.1093/nar/gkt437)
- Chiu et al., [DNAshapeR: an R/Bioconductor package for DNA shape prediction and feature encoding](https://doi.org/10.1093/bioinformatics/btw184)

The feature extractor calls a **local** `Rscript` process and requires the
DNAshapeR package. It does not download sequences or call a hosted service.
Each DNAshapeR feature's native position dimension is recorded separately;
base-pair and base-pair-step quantities are not assumed to share aligned
positions or each have exactly 101 values. Saved batches are right-padded
with NaN only to make variable feature widths rectangular; each feature's
original width is stored alongside the values, and masks distinguish native
values from padding/missing outputs. There is no interpolation or fabricated
value replacement. The raw output tensor is `[sequences, 4, max_native_positions]`.

Feature extraction results are optionally cached as compressed NPZ files,
keyed by sequence and DNAshapeR version. Dataset artifacts preserve sequence,
label, and feature order in separate train/test NPZ files. Preparation fails
if a sequence fails, train/test sequences overlap, or output rows/dimensions
do not align. The training script fits shape normalization on the training
subset only and stores those statistics in the shape-model checkpoint.

## Install and run the DNA-shape pipeline

1. Install R for Windows and ensure `Rscript.exe` is on `PATH`. For a
   non-default install location, set `DNA_SHAPER_RSCRIPT` to its full path.
2. Install DNAshapeR in the R installation used by `Rscript`:

   ```r
   user_library <- Sys.getenv("R_LIBS_USER")
   dir.create(user_library, recursive = TRUE, showWarnings = FALSE)
   .libPaths(c(user_library, .libPaths()))
   if (!requireNamespace("BiocManager", quietly = TRUE))
       install.packages("BiocManager", lib = user_library)
   BiocManager::install(
       "DNAshapeR", lib = user_library, ask = FALSE, update = FALSE
   )
   ```

   Confirm the same R installation can load it:

   ```powershell
   Rscript -e "library(DNAshapeR); packageVersion('DNAshapeR')"
   ```

3. Provide the existing processed files at
   `ml/data/processed/train_clean.csv` and `ml/data/processed/test_clean.csv`.
   Each must contain `sequence` and binary `label` columns (the existing
   preparation also expects `chr_id`).
4. From `ml`, generate the native features:

   ```powershell
   python prepare_shape_dataset.py
   ```

   This writes `data/processed/dna_shape/train_shape.npz` and
   `test_shape.npz`, plus a versioned sequence cache under
   `data/processed/dna_shape_cache/`.
5. Train and compare all four configurations on the same train/validation/test
   split:

   ```powershell
   python train_comparison.py --epochs 30 --batch-size 32
   ```

   It reports accuracy, precision, recall, F1, ROC-AUC, and average
   precision (PR-AUC), and writes `models/dna_shape_comparison_results.json`.
   ROC-AUC is reported as unavailable if the test set has only one class.
   Existing `best_cnn.pth` and `best_hybrid.pth` are not overwritten; the new
   shape model is saved separately as `models/best_shape_fusion.pth`.
6. Run API inference from `ml`:

   ```powershell
   uvicorn api:app --host 127.0.0.1 --port 8001
   ```

   (Port 8001 matches the `/api` proxy in `frontend/vite.config.js`.)

   Send the existing request shape, `{"sequence": "<101 A/C/G/T bases>"}`.
   In the frontend, click **Generate DNA shape matrix** to request the
   `/dna-shape` endpoint and display the native MGW, Roll, ProT, and HelT
   values in a position-by-feature table. This matrix-only action uses
   DNAshapeR and does not require a trained shape-fusion checkpoint. It still
   requires `Rscript` and DNAshapeR to be installed and available to the API
   process; otherwise the endpoint returns an explicit setup error.
   If a compatible `best_shape_fusion.pth` is present, the API computes
   DNAshapeR features locally and uses the sequence-plus-shape model. Without
   that checkpoint, the existing sequence-model selection/fallback remains in
   effect. A shape checkpoint without its matching DNAshapeR installation or
   version fails explicitly; it does not silently fall back to invented shape
   values.

## New model dimensions

For sequence batch size `B`, sequence length `L=101`, and feature-specific
DNAshapeR native widths `P_MGW`, `P_Roll`, `P_ProT`, and `P_HelT`, let
`Pmax = max(P_MGW, P_Roll, P_ProT, P_HelT)`:

| Stage | Tensor shape |
|---|---|
| One-hot sequence input | `[B, 4, 101]` |
| DNAshapeR values (right-padded for batching only) | `[B, 4, Pmax]` |
| One feature's values + validity mask | `[B, 2, Pmax]` |
| Existing tri-branch sequence representation | `[B, 240]` |
| Shape encoder representation | `[B, 64]` |
| Fused sequence/shape representation | `[B, 304]` |
| Binding logit | `[B, 1]` |

Each feature's actual native position count and the batch `Pmax` are recorded
in every feature artifact and model checkpoint. The model encodes each feature
on its own ordered native axis, and does not assume positional alignment
between base-pair and base-pair-step features. DNAshapeR's boundary behavior
is retained as returned by the installed package; no feature is expanded or
shifted to force it to 101 positions.

The processed project data has chromosome IDs but no genomic interval
coordinates. Preparation detects exact duplicate sequences within and across
train/test; it cannot detect overlapping genomic windows using coordinates
that the current dataset does not store. Use a coordinate-aware or
chromosome-held-out split if required by the study protocol.

## Status and limitations

R 4.6.1 and DNAshapeR 1.40.0 were installed and the local predictor was smoke
tested on a valid 101-bp sequence. That run returned `[4, 101]` values with
native feature widths `[101, 100, 101, 100]` in `MGW, Roll, ProT, HelT` order.
This checkout has no processed/raw project dataset files, so dataset feature
generation, model training, and comparative metrics remain unverified. Tests
in `ml/tests` mock the external predictor boundary; they verify validation,
alignment, caching, model tensor shapes, checkpoint compatibility, and the
API handler without claiming mocked values are biological predictions. The
synthetic generator is suitable only for pipeline smoke tests, not a
biological performance claim. This model predicts sequence-context shape only;
it does not account for methylation, protein-induced deformation, chromatin,
or solution conditions.

## Explanation views and DNA tutor

After a prediction, the results area has three views:

- **3D DNA helix** (`frontend/src/components/DnaHelix3D.jsx`): the input sequence drawn as idealised B-DNA. Each base pair is coloured by ShiftSmooth importance, by in-silico-mutation impact (`ism_position_effect` from `/predict`), or by base identity. The dashed lines are the Watson–Crick hydrogen bonds (A–T 2, G–C 3). Click a base pair for its details and an "Ask the tutor" shortcut. It is drawn with plain canvas and redraws only when you interact with it, with no animation loop and no 3D library.
- **2D tracks**: the saliency, ISM and motif panels above.
- **Real protein–DNA structure** (`StructureViewer.jsx`): loads an experimental PDB complex for the matching protein family (1CDW TBP–TATA, 1AAY Zif268 zinc fingers as an SP1-family example, 4AWL NF-Y–CCAAT, 1DH3 CREB–CRE, 5KKQ CTCF). Protein residues within 3.5 Å of the DNA are highlighted. 3Dmol.js is loaded from 3Dmol.org only when this tab is opened, so the tab needs an internet connection. These are reference structures, not predictions for the input sequence.

The **DNA tutor** (`ml/chatbot/`, endpoint `POST /chat`) explains DNA, transcription factors, the model and the user's current result in Beginner or Advanced mode. It answers "why is position N important?" from that prediction's saliency and ISM values. Without an API key it answers offline from `ml/chatbot/knowledge_base.json`. For fuller AI answers, copy `ml/.env.example` to `ml/.env` and add a free Groq or Gemini key. `ml/.env` is git-ignored. Tests: `cd ml && python -m unittest tests.test_chatbot`.

### Running on a laptop

- Train in Google Colab, not on a laptop. The API only needs the saved checkpoints.
- PyTorch is capped at 2 CPU threads (`TORCH_NUM_THREADS`).
- Stop the backend and the frontend with **Ctrl+C**. Nothing keeps running in the background afterwards.
