from fastapi import FastAPI
from fastapi.responses import RedirectResponse, PlainTextResponse
from pydantic import BaseModel
import uvicorn
import sys

from hate.pipeline.train_pipeline import TrainPipeline
from hate.pipeline.prediction_pipeline import PredictionPipeline
from hate.exception import CustomException
from hate.constants import APP_HOST, APP_PORT


app = FastAPI(
    title="Hate Speech Detection API",
    version="1.0.0"
)


# -----------------------------
# Request Model
# -----------------------------
class TextRequest(BaseModel):
    text: str


# -----------------------------
# Home Route
# -----------------------------
@app.get("/", tags=["Home"])
async def home():
    return RedirectResponse(url="/docs")


# -----------------------------
# Training Route
# -----------------------------
@app.get("/train", tags=["Training"])
async def train():
    try:
        train_pipeline = TrainPipeline()
        train_pipeline.run_pipeline()

        return PlainTextResponse(
            content="Training completed successfully!",
            status_code=200
        )

    except Exception as e:
        return PlainTextResponse(
            content=f"Training failed: {str(e)}",
            status_code=500
        )


# -----------------------------
# Prediction Route
# -----------------------------
@app.post("/predict", tags=["Prediction"])
async def predict(request: TextRequest):
    try:
        pipeline = PredictionPipeline()

        prediction = pipeline.run_pipeline(request.text)

        return {
            "input": request.text,
            "prediction": prediction
        }

    except Exception as e:
        raise CustomException(e, sys)


# -----------------------------
# Run Application
# -----------------------------
if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=True
    )