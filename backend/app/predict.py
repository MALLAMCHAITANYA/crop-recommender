import os
import joblib
import pandas as pd

# Base directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load trained model and label encoder
model = joblib.load(os.path.join(BASE_DIR, "model.pkl"))
label_encoder = joblib.load(os.path.join(BASE_DIR, "label_encoder.pkl"))

# Feature names (MUST MATCH TRAINING)
FEATURES = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']


def predict_top3(input_data):
    """
    input_data: [[N, P, K, temperature, humidity, ph, rainfall]]
    """

    # ✅ Convert input to DataFrame (THIS FIXES THE ERROR)
    df = pd.DataFrame(input_data, columns=FEATURES)

    # Get probability predictions
    probabilities = model.predict_proba(df)[0]

    # Get top 3 indices
    top3_indices = probabilities.argsort()[-3:][::-1]

    results = []
    for idx in top3_indices:
        crop_name = label_encoder.inverse_transform([idx])[0]
        confidence = round(probabilities[idx] * 100, 2)

        results.append({
            "crop": crop_name,
            "confidence": confidence
        })

    return results
