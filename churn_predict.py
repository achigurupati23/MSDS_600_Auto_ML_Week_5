import pandas as pd
from pycaret.classification import load_model, predict_model

_model = load_model("week5_churn_model")

def predict_churn(csv_path: str):
    df = pd.read_csv(csv_path)
    preds = predict_model(_model, data=df)
    prob_col = "prediction_score" if "prediction_score" in preds.columns else None
    cols = ["prediction_label"] + ([prob_col] if prob_col else [])
    print(preds[cols].head(10))
    return preds
