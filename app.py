from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import os
from pathlib import Path

from models import chest_xray, brain_ct, mammography, msk_fracture, ultrasound

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dynamic model selector
model_map = {
    "chest_xray": chest_xray,
    "brain_ct": brain_ct,
    "mammography": mammography,
    "msk_fracture": msk_fracture,
    "ultrasound": ultrasound
}

@app.post("/predict")
async def predict_image(modality: str = Form(...), file: UploadFile = File(...)):
    if modality not in model_map:
        return JSONResponse(status_code=400, content={"error": "Invalid modality."})

    model_module = model_map[modality]
    image_bytes = await file.read()

    try:
        label, tensor, model, target_layer = model_module.predict(image_bytes)
        # Here you can use Grad-CAM based on model & target_layer if available
        return {"modality": modality, "prediction": label}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/")
def home():
    return {"message": "MedScanAI Backend is live"}
