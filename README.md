# Calibration and Reliability of RAG Verification Systems under Evidence Perturbations

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Abstract
*Placeholder: Insert research abstract here.*

Retrieval-Augmented Generation (RAG) systems rely critically on verification components to assess the factual consistency and relevance of retrieved evidence against generated claims. However, little is known regarding the confidence calibration and reliability of these verification mechanisms under systematic evidence perturbations (e.g., entity swapping, negation insertion, distractor passages, and temporal shifts). This study investigates the calibration characteristics (Expected Calibration Error, reliability diagrams, temperature scaling) of RAG verification modules across perturbed context distributions to assess failure modes and improve out-of-distribution robustness.

---

## Project Structure

```text
rag-verification-calibration/
├── data/                  # Data loading scripts (no raw data files tracked)
├── notebooks/             # Colab and Jupyter exploratory notebooks
├── report/                # Research paper draft, LaTeX source, or writing
├── results/               # Experimental outcomes
│   ├── figures/           # Generated calibration curves & plots
│   ├── raw_logs/          # Raw run logs and inference traces
│   └── tables/            # Formatted performance and ECE metric tables
├── src/                   # Core research source code
│   ├── __init__.py
│   ├── calibration.py     # Calibration metrics (ECE, MCE, reliability diagrams, Platt/temperature scaling)
│   ├── data_loader.py     # Dataset ingestion and preprocessing pipelines
│   ├── evaluate.py        # End-to-end evaluation routines and reporting
│   ├── perturbations.py   # Systematic evidence perturbation operators
│   └── train_verifier.py # Training and fine-tuning routines for verification models
├── .gitignore             # Ignore data, checkpoints, venvs, and notebooks checkpoints
├── README.md              # Project documentation
└── requirements.txt       # Project dependencies
```

---

## Setup Instructions

### 1. Prerequisites
- Python 3.10 or higher
- (Optional) CUDA-compatible GPU for accelerated inference and training

### 2. Environment Setup

Create and activate a virtual environment:

```bash
# On Linux/macOS
python3 -m venv venv
source venv/bin/activate

# On Windows (PowerShell)
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

Install the required packages listed in `requirements.txt`:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. API Keys (If Applicable)

If running evaluations with Gemini or other hosted LLMs:

```bash
export GEMINI_API_KEY="your-api-key-here"
```

---

## Pipeline Overview

1. **Data Preparation**: Prepare verification datasets using [`src/data_loader.py`](src/data_loader.py).
2. **Evidence Perturbation**: Generate synthetic distractors, corrupted passages, and counterfactual evidence using [`src/perturbations.py`](src/perturbations.py).
3. **Verifier Training**: Train or fine-tune verification models with [`src/train_verifier.py`](src/train_verifier.py).
4. **Calibration**: Measure and adjust confidence scores (Temperature Scaling, Platt Scaling) with [`src/calibration.py`](src/calibration.py).
5. **Evaluation**: Benchmark verification reliability and export tables and figures via [`src/evaluate.py`](src/evaluate.py).

---

## Results

*Placeholder: Experimental findings, benchmark comparisons, and calibration plots will be reported here.*

- **Expected Calibration Error (ECE)** under standard vs. perturbed evidence
- **Selective Classification AUC** & Risk-Coverage profiles
- **Impact of Post-hoc Calibration** methods across perturbational regimes

---

## Citation

```bibtex
@misc{rag_verification_calibration,
  title={Calibration and Reliability of RAG Verification Systems under Evidence Perturbations},
  author={Dixit, Adarsh},
  year={2026}
}
```
