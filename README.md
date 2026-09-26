# Investment Risk Profiler

A machine learning app that classifies investors into risk tolerance
categories — **Conservative**, **Moderate**, or **Aggressive** — based on
their financial and demographic profile, with real-time predictions served
through an interactive Streamlit interface.

## Tech Stack
Python • Streamlit • Scikit-learn • Pandas • NumPy

## Project Structure
```
investment_risk_app/
├── generate_data.py    # Creates the synthetic investor dataset
├── train_model.py       # Preprocessing, training, evaluation, saves model.pkl
├── app.py                # Streamlit web app for real-time predictions
├── requirements.txt
└── README.md
```

## Setup & Usage

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Train the model (creates `investor_data.csv` and `model.pkl`):
   ```bash
   python train_model.py
   ```

3. Launch the web app:
   ```bash
   streamlit run app.py
   ```

4. Open the local URL Streamlit prints (usually `http://localhost:8501`),
   enter an investor profile, and click **Predict Risk Profile**.

## Model Details

- **Algorithm:** RandomForestClassifier (200 trees, max depth 10, balanced
  class weights)
- **Features:** Age, Annual Income, Investment Amount, Years of Experience,
  Financial Knowledge Score, Portfolio Diversity Score, Debt-to-Income Ratio
- **Preprocessing:** StandardScaler feature scaling, LabelEncoder for the
  target classes, stratified 80/20 train/test split
- **Evaluation:** ~**85.7% test accuracy**, ~84.9% 5-fold cross-validated
  accuracy, plus a full classification report (precision/recall/F1) and
  confusion matrix printed by `train_model.py`

## Resume Bullet

> Built and deployed a machine learning model achieving **86% accuracy** in
> classifying investors into risk tolerance categories, using a Streamlit
> web app for real-time, interactive predictions. Applied data
> preprocessing, feature scaling, and model evaluation (cross-validation,
> precision/recall/F1) to select and tune a RandomForest classifier.
