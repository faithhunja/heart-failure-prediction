# Import required packages
import pickle
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline


# Resolve paths from this script so training works from any current directory.
DATA_FILE = Path(__file__).parent / "data" / "heart.csv"
OUTPUT_FILE = Path(__file__).parent / "model.bin"
TARGET = "HeartDisease"

# These columns must match the features accepted by the prediction API.
CATEGORICAL = [
    "Sex",
    "ChestPainType",
    "RestingECG",
    "ExerciseAngina",
    "ST_Slope",
]
NUMERICAL = [
    "Age",
    "RestingBP",
    "Cholesterol",
    "FastingBS",
    "MaxHR",
    "Oldpeak",
]


def load_data():
    return pd.read_csv(DATA_FILE)


def train_model(df):
    features = CATEGORICAL + NUMERICAL
    train_dict = df[features].to_dict(orient="records")
    model = make_pipeline(
        DictVectorizer(sparse=False),
        LogisticRegression(solver="liblinear", C=1.0, max_iter=1000),
    )
    model.fit(train_dict, df[TARGET])
    return model


def save_model(model, output_file):
    # Save the fitted vectorizer and classifier together for prediction.
    with open(output_file, "wb") as file:
        pickle.dump(model, file)


# Train on the dataset, then save the pipeline used by predict.py.
df = load_data()
model = train_model(df)
save_model(model, output_file=OUTPUT_FILE)

print(f'Model saved to {OUTPUT_FILE}')
