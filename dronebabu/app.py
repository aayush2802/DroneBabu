from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import pickle
import os

app = FastAPI()

# Load the pre-trained KNN model
model_path = os.path.join(os.getcwd(), "knn_model.pkl")
with open(model_path, "rb") as f:
    model = pickle.load(f)

# Define the input schema
class CropInput(BaseModel):
    Carbon: float
    Organic_Matter: float
    Phosphorous: float
    Calcium: float
    Magnesium: float
    Potassium: float

@app.post("/predict/")
async def predict_crop(input: CropInput):
    try:
        # Prepare the feature array for prediction
        features = np.array([[input.Carbon, input.Organic_Matter, input.Phosphorous,
                               input.Calcium, input.Magnesium, input.Potassium]])
        # Make a prediction
        predicted_crop = model.predict(features)[0]

        # Map the prediction to crop names
        crop_name = "Soybean" if predicted_crop == 1 else "Paddy"

        # Return the prediction
        return {"predicted_crop": crop_name}

    except Exception as e:
        return {"error": str(e)}