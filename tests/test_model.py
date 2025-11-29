import os
from src.model import train_model, load_model, MODEL_PATH, predict_price


def test_train_model_creates_file():
    score = train_model()

    # model file should be created
    assert os.path.exists(MODEL_PATH)

    # simple check on score range
    assert 0.0 <= score <= 1.0


def test_predict_price_runs():
    model = load_model()
    assert model is not None

    price = predict_price(size=1200, bedrooms=3, age=10)
    # price should be a positive float
    assert isinstance(price, float)
    assert price > 0
