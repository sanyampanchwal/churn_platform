# B2B Employee Retention Intelligence Platform

An end-to-end Machine Learning data pipeline and dashboard for predicting and analyzing corporate **Employee Attrition (HR)**. This project demonstrates highly novel B2B full-stack data science capabilities, encompassing data ingestion, exploratory data analysis (EDA), predictive modeling with a Random Forest classifier, and an interactive, aesthetically premium Plotly-Dash visualization layer.

## Platform Features

- **Novel Academic Dataset**: Uses the official IBM HR Analytics Employee Attrition dataset, proving adaptability beyond over-used consumer data.
- **Automated EDA**: Built-in statistical correlations and Kaplan-Meier survival curves to chart employee resignation timelines natively.
- **Machine Learning Classification**: A heavily tuned `RandomForestClassifier` operating on 35 distinct internal HR features (income, commute, overtime) to achieve robust ROC-AUC targets and flag "flight-risk" talent.
- **SHAP Game-Theory Explainability**: Instantly unlocks the "Black Box", letting HR directors know exactly *why* algorithms flag specific talent to quit.
- **Aesthetic Premium UI**: A beautiful, dark-themed interactive web app featuring responsive KPI contextual cards, custom visual tooltips, and dynamic dataframe slice clustering in real-time.

## Quick Start Guide

### 1. Installation
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run the Full Pipeline
Execute `main.py` to securely fetch the IBM dataset, run statistical survival analytics, train the AI, and inject predictions:
```bash
python main.py
```

### 3. Launch the Dashboard
```bash
python dashboard/dashboard.py
```
Navigate to [http://127.0.0.1:8050](http://127.0.0.1:8050) in your web browser to explore.
