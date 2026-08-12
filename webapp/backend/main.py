"""
学習済みモデルで天気図画像を分類するFastAPI推論サーバー。

起動:
    uvicorn webapp.backend.main:app --reload --port 8000
"""

import io
import os

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from src.labels import INDEX_TO_LABEL, LABELS
from src.model import build_model
from src.train import get_transforms

WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", "weights/model.pt")

app = FastAPI(title="Weather Pattern Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = get_transforms(train=False)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(WEIGHTS_PATH):
        # モデル未学習の段階でもAPIサーバー自体は起動できるようにしておく
        print(f"warning: weights not found at {WEIGHTS_PATH}. /predict will fail until trained.")
        return
    m = build_model(num_classes=len(LABELS))
    m.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
    m.eval()
    m.to(device)
    model = m


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet. Train and place weights first.")

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = int(torch.argmax(probs).item())
    return {
        "label": INDEX_TO_LABEL[top_idx],
        "confidence": float(probs[top_idx]),
        "all_probabilities": {INDEX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)},
    }
