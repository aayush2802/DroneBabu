from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import joblib
import ctypes  # Replacing pywin32 with ctypes for Windows API interaction

app = FastAPI()

# Load the pre-trained KNN model
model = joblib.load('knn_model.pkl')

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

        # Make a prediction using the pre-trained KNN model
        predicted_crop = model.predict(features)[0]

        # Map the prediction to crop names
        if predicted_crop == 1:
            crop_name = "Soybean"
        else:
            crop_name = "Paddy"

        # OPTIONAL: Use ctypes for Windows-specific functionality (e.g., show a MessageBox)
        ctypes.windll.user32.MessageBoxW(
            None,
            f"Predicted crop: {crop_name}",
            "Prediction Alert",
            1
        )

        # Return the crop name as a string in the response
        return {"predicted_crop": crop_name}

    except Exception as e:
        return {"error": str(e)}
