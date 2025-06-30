# backend.py
import pandas as pd
import joblib

# Load model once
model = joblib.load("C:/Users/viswa/Desktop/IITI/Digestor Project/penaltyMODEL/data/trained_model.joblib")

def predict_methane_yield(input_dict):
    df = pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]
    return round(prediction, 4)
