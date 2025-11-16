import pickle
from typing import Any, Dict

import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException
from sklearn.feature_extraction import DictVectorizer

from schemas import PredictionResponse, WaterSample

# Loading the model with Pickle
model_file = "model_C=1.0.bin"

dv = DictVectorizer(sparse=True)

app = FastAPI(title="Water Potability Prediction API")


with open(model_file, "rb") as f_in:
    model = pickle.load(f_in)


@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionResponse)
def predict_potability(sample: WaterSample):
    try:
        X = dv.fit_transform(sample.model_dump())

        # Predict
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0]

        # Generate recommendation
        if prediction == 1:
            recommendation = (
                "Water appears safe for consumption based on chemical analysis."
            )
        else:
            recommendation = (
                "Water quality concerns detected. Further testing recommended."
            )

        return {
            "potable": bool(prediction),
            "confidence": float(probability[1]),
            "recommendation": recommendation,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {
        "message": "Water Potability Prediction API",
        "endpoints": {
            "/predict": "POST - Predict water potability",
            "/health": "GET - Health check",
            "/docs": "GET - API documentation",
        },
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9696)
