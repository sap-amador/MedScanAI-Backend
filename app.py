from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from models.predict import predict_image_from_modality

app = FastAPI()

# CORS setup for frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def predict(modality: str = Form(...), file: UploadFile = File(...)):
    result = await predict_image_from_modality(modality, file)
    return result
