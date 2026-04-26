# AI-Based Crop Yield Prediction and Advisory System

A deep learning application that predicts crop yield from time-series agricultural features using an **LSTM** model, and delivers **farmer-facing advisory** guidance. The interactive experience is built with **Streamlit**; training and evaluation use **PyTorch**.

---

## 1. Project Overview

This project implements an end-to-end workflow: **partial-season time series** are fed into an LSTM to estimate yield, results are combined with **regional context** (including agro-climatic zone–based rules in the UI), and outputs include **yield ranges** to reflect uncertainty. A separate rule-based layer provides **actionable advisory** (irrigation, monitoring, zone-specific tips) alongside the model-driven estimate.

---

## 2. Features

| Feature | Description |
|--------|-------------|
| **LSTM yield prediction** | Sequence model (`torch`) trained on aligned `.npy` feature tensors and yield labels. |
| **Agro-climatic zoning** | Indian states mapped to zones (Coastal, Semi-arid, Tropical) for contextual recommendations—separate from core tensor input. |
| **Uncertainty-aware display** | Yield shown as a **range** (e.g. low–high t/ha) with week-dependent margin, not only a single point. |
| **Advisory system** | `advisory.py` classifies predictions vs. regional average; the app adds weekly focus tasks, weather copy, and zone-specific panels. |
| **Evaluation pipeline** | `evaluate.py` produces metrics, CSVs, and diagnostic plots under `results/`. |

---

## 3. Tech Stack

| Layer | Technology |
|-------|------------|
| Language | **Python 3** |
| Deep learning | **PyTorch** (LSTM, `torch.utils.data`) |
| App / UI | **Streamlit**, **Plotly** |
| Numerics & data | **NumPy** |
| Visualization (offline eval) | **Matplotlib** |

> **Note:** The implementation uses **PyTorch**, not TensorFlow/Keras. If you extend the project with Keras models, update this section accordingly.

---

## 4. Project Structure

```
crop-yield-prediction/
├── app.py                 # Streamlit web app (main user-facing entry)
├── train.py               # GPU training script; saves weights to models/model.pth
├── evaluate.py            # Test-set metrics + plots + CSV exports
├── advisory.py            # Yield vs. average advisory logic
├── data_loader.py         # Optional: inspect raw X_timeseries / y_yield shapes
├── gpu_check.py           # Optional: quick CUDA sanity check
├── requirements.txt       # Direct third-party dependencies
├── data/
│   ├── X_final.npy        # Aligned feature tensor (used by app, train, eval)
│   ├── y_final.npy        # Yield targets (aligned with X_final)
│   ├── X_timeseries.npy   # Raw / upstream series (preprocessing input)
│   ├── y_yield.npy        # Raw yield vector (preprocessing input)
│   ├── torch_dataset.py   # PyTorch Dataset (variable-length / partial-week sampling)
│   └── align_dataset.py   # One-off alignment: builds X_final / y_final from raw arrays
├── models/
│   ├── lstm_yield.py      # LSTM architecture + packed-sequence handling
│   └── model.pth          # Trained weights (created by train.py)
└── results/               # Outputs from evaluate.py (CSVs, PNGs)
```

---

## 5. Installation Instructions

### Clone the repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd crop-yield-prediction
```

### Create a virtual environment

```bash
python -m venv venv
```

**Activate the environment**

- **Linux / macOS**

  ```bash
  source venv/bin/activate
  ```

- **Windows (CMD)**

  ```cmd
  venv\Scripts\activate
  ```

- **Windows (PowerShell)**

  ```powershell
  venv\Scripts\Activate.ps1
  ```

### Install dependencies

```bash
pip install -r requirements.txt
```

**GPU (optional):** Training in `train.py` expects CUDA. Install a **CUDA-enabled** PyTorch build from the [official PyTorch install guide](https://pytorch.org/get-started/locally/) if the default `pip install torch` CPU wheel is not sufficient.

---

## 6. How to Run the Project

### Web application (recommended entry)

From the project root, with the virtual environment activated:

```bash
streamlit run app.py
```

The app loads:

- `models/model.pth` — trained LSTM weights  
- `data/X_final.npy`, `data/y_final.npy` — feature tensor and labels  

If these files are missing, the UI will show an error and stop.

### Train the model (GPU)

```bash
python train.py
```

Requires CUDA (`train.py` raises if GPU is unavailable). Writes `models/model.pth`.

### Evaluate and regenerate result artifacts

```bash
python evaluate.py
```

Uses the same dataset paths as training and writes under `results/` (see below).

### Optional utilities

```bash
python data_loader.py      # Inspect raw .npy shapes (development)
python gpu_check.py        # Verify CUDA / GPU visibility
python data/align_dataset.py   # Rebuild X_final.npy / y_final.npy from X_timeseries.npy + y_yield.npy
```

---

## 7. Important Files to Check

| File | Role |
|------|------|
| **`app.py`** | Main Streamlit entry: model load, prediction UI, agro-climatic mapping, advisory copy. |
| **`models/lstm_yield.py`** | LSTM model definition (equivalent role to a standalone `model.py`). |
| **`data/torch_dataset.py`** | Dataset + partial-week sampling for training/evaluation. |
| **`data/align_dataset.py`** | Preprocessing script to align and save `X_final.npy` / `y_final.npy`. |
| **`train.py` / `evaluate.py`** | Training loop and evaluation + file exports. |
| **`advisory.py`** | Advisory thresholds and messages from predicted vs. average yield. |
| **`data/*.npy`** | Required inputs for app and training; keep paths consistent with scripts. |

---

## 8. Output Explanation

### In the Streamlit app

- **Yield range** (t/ha): displayed band around the central estimate.  
- **Crop condition** / health category: derived from predicted yield vs. regional average.  
- **Weekly focus**, **zone recommendations**, and **weather-style** panels: rule-based UX layers.

### From `evaluate.py` (`results/`)

| Output | Description |
|--------|-------------|
| `evaluation_metrics.csv` | Aggregated error metrics (e.g. MAE, RMSE). |
| `predicted_vs_actual_yield.csv` | Per-sample actual vs. predicted yield (and error). |
| `actual_vs_predicted_scatter.png` | Scatter of predicted vs. actual. |
| `error_distribution_histogram.png` | Distribution of prediction errors. |
| `actual_vs_predicted_lineplot.png` | Example index-wise comparison for a subset of samples. |

**Example:** Open `predicted_vs_actual_yield.csv` in a spreadsheet or notebook to compare columns `Actual_Yield` and `Predicted_Yield` for each `Sample_Index`.

---

## 9. Screenshots / Demo

Add screenshots to your repository (e.g. `docs/images/`) and link them here:

```markdown
![Streamlit home](docs/images/streamlit-home.png)
```

If you record a short demo video, link it from this section (e.g. Loom or YouTube).

---

## 10. Future Improvements

- **Model:** Experiment with attention, Transformer encoders, or multi-crop heads; add proper validation split and early stopping.  
- **Data:** Document provenance of `.npy` files; version datasets with DVC or similar.  
- **Uncertainty:** Replace heuristic range with MC dropout, ensembles, or quantile regression.  
- **MLOps:** Pin dependency versions in `requirements.txt`, add CI, and containerize with Docker.  
- **Product:** Optional login, localization, and integration with real weather/soil APIs (with clear separation from model inputs).  
- **Tests:** Unit tests for `advisory.py` and shape contracts on the dataset module.

---

## 11. Author Details

| | |
|---|---|
| **Author** | *[Your Name]* |
| **Contact** | *[Your email or portfolio URL]* |
| **License** | *[e.g. MIT — add a LICENSE file if applicable]* |

---

*Built for learning and portfolio demonstration. Predictions and advice are illustrative; always validate agronomic decisions with local experts and field data.*
