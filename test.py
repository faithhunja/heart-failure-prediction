# Import required packages
import requests

url = "http://localhost:9696/predict"

# Example patient for testing the prediction model
patient = {
    "Age": 40,
    "Sex": "M",
    "ChestPainType": "ATA",
    "RestingBP": 140,
    "Cholesterol": 289,
    "FastingBS": 0,
    "RestingECG": "Normal",
    "MaxHR": 172,
    "ExerciseAngina": "N",
    "Oldpeak": 0.0,
    "ST_Slope": "Up",
}

response = requests.post(url, json=patient, timeout=10)
# Raise an error for failed HTTP responses before parsing the prediction
response.raise_for_status()
prediction = response.json()

print(f"Heart disease probability: {prediction['heart_disease_probability']:.3f}")
if prediction["heart_disease"]:
    print("The model predicts heart disease.")
else:
    print("The model predicts no heart disease.")
