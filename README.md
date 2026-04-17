# Customer Churn Intelligence Platform

An end-to-end Machine Learning data pipeline and dashboard for predicting and analyzing customer churn. This project demonstrates strong full-stack data science capabilities, encompassing data generation, exploratory data analysis (EDA), predictive modeling with a Random Forest classifier, and an interactive Plotly-Dash visualization layer.

## Platform Features

- **Robust Data Pipeline**: Programmatic generation of realistic multi-featured telecommunication datasets.
- **Automated EDA**: Built-in statistical correlations and Kaplan-Meier survival curves using `lifelines` and `seaborn`.
- **Machine Learning**: A `RandomForestClassifier` heavily tuned with class weighting to achieve reliable ROC-AUC boundaries.
- **SHAP Explainability**: Integrates Shapley Additive Explanations for real-time model interpretability preventing black-box insights.
- **Interactive Web App**: A dynamic React-based `Dash` frontend offering KPI cards, data slice filtering, and distribution interactivity.

## Architecture

```text
churn_platform/
├── main.py                  # Pipeline Orchestrator
├── requirements.txt         # Dependency Manifest
├── data/
│   └── generate_data.py     # Data Simulator / Ingestion Module
├── eda/
│   └── analysis.py          # Statistics & Plot Generation
├── model/
│   └── train_model.py       # Random Forest & SHAP Evaluator
├── dashboard/
│   └── dashboard.py         # Dash Plotly Web Server
└── outputs/                 # Output repository (Models, CSVs, Visuals)
```

## Quick Start Guide

### 1. Installation

Clone this repository and set up your Python environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On macOS/Linux:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Run the Full Analytics Pipeline
Execute `main.py` which will sequentially generate data, run analytics, train the ML model, and save all outputs:

```bash
python main.py
```
*Wait until it prints: `✓ All pipeline steps complete.`*

### 3. Launch the Dashboard
Serve the interactive visualizer on your local machine:

```bash
python dashboard/dashboard.py
```
Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) in your web browser to explore.
