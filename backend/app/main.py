from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

app = FastAPI()  # <-- Create app first!

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model and encoder
model = joblib.load("app/model.pkl")
label_encoder = joblib.load("app/label_encoder.pkl")

@app.post("/predict")
async def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df)[0]
    crop = label_encoder.inverse_transform([prediction])[0]
    return {"Recommended Crop": crop}

@app.get("/")
def home():
    return {"message": "Crop Recommendation API Running!"}
