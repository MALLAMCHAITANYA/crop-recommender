from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.predict import predict_top3
from app.schemas import CropInput, PredictionResponse

app = FastAPI(title="Crop Recommendation System")

# -------------------------------------------------
# Enable CORS
# -------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------
# Crop Prediction API (Top-3)
# -------------------------------------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(data: CropInput):

    input_data = [[
        data.N,
        data.P,
        data.K,
        data.temperature,
        data.humidity,
        data.ph,
        data.rainfall
    ]]

    results = predict_top3(input_data)

    return {
        "recommendations": results
    }

# -------------------------------------------------
# Root Endpoint
# -------------------------------------------------
@app.get("/")
def home():
    return {"message": "Crop Recommendation API Running!"}
