from fastapi import FastAPI
from pydantic import BaseModel
from .model import load_model

app = FastAPI(title="House Price Prediction API", version="1.0.0")

model = load_model()


class HouseFeatures(BaseModel):
    size: float       # in square feet
    bedrooms: int
    age: float        # in years


@app.get("/")
def root():
    return {"message": "House Price Prediction API - staging"}


@app.post("/predict")
def predict_price(features: HouseFeatures):
    import numpy as np

    X = np.array([[features.size, features.bedrooms, features.age]])
    price = float(model.predict(X)[0])
    return {"predicted_price": price}
