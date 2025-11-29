import os
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "house_price_model.pkl")


def generate_synthetic_data(n_samples: int = 300, random_state: int = 42):
    """
    Create a fake house-price dataset:
    features: [size_sqft, bedrooms, age_years]
    """
    rng = np.random.RandomState(random_state)

    size = rng.randint(500, 3500, size=n_samples)       # square feet
    bedrooms = rng.randint(1, 6, size=n_samples)        # 1–5
    age = rng.randint(0, 40, size=n_samples)            # 0–40 years

    X = np.column_stack([size, bedrooms, age])

    # simple price formula + noise
    noise = rng.normal(0, 10000, size=n_samples)
    y = 50_000 + size * 150 + bedrooms * 10_000 - age * 1_000 + noise

    return X, y


def train_model():
    """
    Train a simple Linear Regression model and save it to disk.
    Returns the R^2 score on the test set.
    """
    X, y = generate_synthetic_data()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = LinearRegression()
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)

    return score


def load_model():
    """
    Load the trained model; if it doesn't exist, train first.
    """
    if not os.path.exists(MODEL_PATH):
        train_model()
    model = joblib.load(MODEL_PATH)
    return model


def predict_price(size: float, bedrooms: int, age: float) -> float:
    """
    Make a price prediction for a single house.
    """
    model = load_model()
    X = np.array([[size, bedrooms, age]])
    pred = model.predict(X)[0]
    return float(pred)


if __name__ == "__main__":
    score = train_model()
    print(f"Model trained. R^2 score: {score:.3f}")
