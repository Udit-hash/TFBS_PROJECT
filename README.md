# TFBS Prediction & Interpretability Workbench

Predicts whether a **101 bp DNA sequence** contains a transcription factor binding site, and explains why.

- **Models:** the main **CNN + BiLSTM + Transformer** model, compared against a CNN, a CNN + BiLSTM, a k-mer logistic regression and a JASPAR motif (PWM) scan.
- **Data:** ENCODE ChIP-seq for the transcription factor **FOSL2** in HepG2 cells, from the [BERT-TFBS](https://github.com/ZX1998-12/BERT-TFBS) benchmark. About 39k training and 10k test sequences. Bound sites are labelled 1; shuffled copies are labelled 0.
- **Explanations:** ShiftSmooth saliency, in-silico mutagenesis (ISM), a scan of the sequence against ~900 known **JASPAR** motifs, a 95% interval for the probability (Monte-Carlo dropout), a **3D DNA helix** coloured by importance, and a **real protein–DNA structure** viewer (e.g. Fos–Jun/AP-1 for FOSL2).
- **What the model learned:** sequence logos from its CNN filters, matched to JASPAR, plus a test of which explanation method finds the real binding site best (`ml/interpret.py`).
- **DNA tutor chatbot:** explains DNA, transcription factors, the model and your own result in Beginner or Advanced mode.

The whole project uses **Python** (model and backend) and **JavaScript/React** (website).

## Run the website (two terminals)

**1. Backend** (from the `ml` folder)

```bash
cd ml
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn api:app --host 127.0.0.1 --port 8001
```

**2. Frontend** (from the `frontend` folder)

```bash
cd frontend
npm install
npm run dev        # open the printed http://localhost:5173 link
```

The website reaches the backend through `/api`, which `frontend/vite.config.js` forwards to port 8001.

**To stop:** press **Ctrl+C** in both terminals. Nothing keeps running afterwards.

**Laptop-friendly:** the website only *uses* trained models; it never trains. PyTorch is limited to 2 CPU threads (`TORCH_NUM_THREADS`). The 3D view only redraws when you interact with it.

## Train the models (Google Colab)

Training happens in Google Colab, not on laptops. Open `notebooks/train_all_models_colab.ipynb` and follow the steps. It:

1. downloads the FOSL2 dataset and cleans it with `ml/prepare_dataset.py`;
2. downloads the JASPAR motif database (`ml/motifs.py`);
3. trains all five methods on **identical data** with **5-fold cross-validation** (`ml/train_all.py`);
4. picks the yes/no cut-off on validation data and scores the untouched test set (AUROC, PR-AUC, accuracy, precision, recall, F1, MCC, ROC curves);
5. explains the trained model: filter logos + JASPAR matches + explanation-method comparison (`ml/interpret.py`);
6. checks the website backend with the new model, and zips everything as `TFBS_results.zip`.

To use the new main model in the website, copy `final_models/tri_branch.pth` into `ml/models/` as `best_hybrid.pth` (the tuned cut-off is stored inside it), and `jaspar_core_vertebrates.json` into `ml/motifs/`.

**More ENCODE datasets:** `ml/build_encode_dataset.py` rebuilds any of the 165 benchmark datasets listed in `ml/datasets_165.txt` from the original ENCODE peak files (UCSC, hg19). Run it in Colab, not on a laptop (it downloads the 800 MB genome).

`ml/make_synthetic_sequences.py` creates **made-up** sequences for the mentor's random-input sanity check. It writes to `ml/data/synthetic/` and is never used for training or reported results. `ml/train.py` is the original single-split trainer, kept for reference; `train_all.py` replaces it.

## DNA tutor chatbot (optional API key)

It works without any key, answering from `ml/chatbot/knowledge_base.json`. For fuller AI answers, copy `ml/.env.example` to `ml/.env` and add a free Groq or Google Gemini key. `ml/.env` is never committed.

## Project layout

```
ml/
  api.py                  backend: /predict (model + explanations) and /chat (tutor)
  train_all.py            trains and compares all 5 methods (5-fold CV)
  interpret.py            filter logos, JASPAR matching, explanation-method comparison
  xai.py                  explanation methods (gradient, ShiftSmooth, Integrated Gradients, MC dropout)
  motifs.py, baselines.py JASPAR tools, PWM and k-mer baselines
  data_utils.py           data loading, dinucleotide shuffle, genome reader
  build_encode_dataset.py rebuild other ENCODE datasets
  predict.py              command-line prediction (one sequence or a FASTA file)
  prepare_dataset.py      cleans train.csv / test.csv
  models/                 cnn.py, cnn_bilstm.py, hybrid.py, loader.py (+ trained .pth files)
  chatbot/                tutor logic and knowledge base
  tests/                  automated tests
frontend/src/
  App.jsx                 page layout, input, results
  components/             3D helix, structure viewer, chat panel
notebooks/
  train_all_models_colab.ipynb
```

## Notes

- The 3D helix is an idealised DNA model built from the input sequence. The colours show the model's importance scores, not measured atomic contacts.
- The structure tab loads lab-measured protein–DNA structures from the Protein Data Bank (1FOS, 1DH3, 1CDW, 1AAY, 4AWL, 5KKQ) as reference examples. It needs internet access.
- DNA shape features (mentioned in the paper) are left for future work; the project is Python + JavaScript only.
- `results/run1_fosl2/` holds the first Colab run (tables, ROC curves, filter logos). `ml/motifs/jaspar_core_vertebrates.json` is JASPAR 2024 CORE vertebrates (open access, CC BY 4.0).
- `docs/PROJECT_EXPLAINED.md` explains every part of the project in plain language (useful for the viva).
- Tests: `cd ml && python -m unittest discover tests`
