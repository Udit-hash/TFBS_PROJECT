# TFBS Prediction & Interpretability Workbench

Predicts whether a **101 bp DNA sequence** contains a transcription factor binding site, and explains why.

- **Models:** CNN (baseline), CNN + BiLSTM, and the main **CNN + BiLSTM + Transformer** model.
- **Data:** ENCODE ChIP-seq for the transcription factor **FOSL2** in HepG2 cells, from the [BERT-TFBS](https://github.com/ZX1998-12/BERT-TFBS) benchmark. About 39k training and 10k test sequences. Bound sites are labelled 1; shuffled copies are labelled 0.
- **Explanations:** ShiftSmooth saliency, in-silico mutagenesis (ISM), the peak motif window, a **3D DNA helix** coloured by importance, and a **real protein–DNA structure** viewer.
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
2. trains all three models on **identical data** with **5-fold cross-validation** (`ml/train_all.py`);
3. picks the yes/no cut-off on validation data and scores the untouched test set;
4. writes a results table (mean ± SD over 5 folds) for the paper, plus the trained model files.

To use the new main model in the website, copy `final_models/tri_branch.pth` into `ml/models/` as `best_hybrid.pth`.

`ml/fetch_encode_data.py` creates **made-up** sequences. It's only for demos and quick tests (for example, showing the mentor a new sequence). Never use it for training or reported results.

## DNA tutor chatbot (optional API key)

It works without any key, answering from `ml/chatbot/knowledge_base.json`. For fuller AI answers, copy `ml/.env.example` to `ml/.env` and add a free Groq or Google Gemini key. `ml/.env` is never committed.

## Project layout

```
ml/
  api.py                  backend: /predict (model + explanations) and /chat (tutor)
  train_all.py            trains and compares the 3 models (5-fold CV)
  prepare_dataset.py      cleans train.csv / test.csv
  models/                 cnn.py, cnn_bilstm.py, hybrid.py (+ trained .pth files)
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
- The structure tab loads lab-measured protein–DNA structures from the Protein Data Bank (1CDW, 1AAY, 4AWL, 1DH3, 5KKQ) as reference examples. It needs internet access.
- Tests: `cd ml && python -m unittest discover tests`
