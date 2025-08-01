from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow frontend to access backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict_image(modality: str = Form(...), file: UploadFile = File(...)):
    # Example logic
    from models.predict import predict_image_from_modality
    result = await predict_image_from_modality(modality, file)
    return result
