"""
学習済みモデルで天気図画像を分類するFastAPI推論サーバー。

起動:
    uvicorn webapp.backend.main:app --reload --port 8000
"""

import io
import os
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from scripts.preprocess_jma import DEFAULT_STAMP_BOX, autocrop_to_content, mask_stamp_box
from src.labels import INDEX_TO_LABEL, LABEL_JA, LABELS
from src.model import build_model
from src.train import get_transforms

WEIGHTS_PATH = os.environ.get("MODEL_WEIGHTS", "weights/model.pt")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(title="Weather Pattern Classification API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ColabのポートフォワーディングだとオリジンがCORSで面倒なため、
# フロントエンドも同じFastAPIプロセスから配信する(相対パスでfetchできる)
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


@app.get("/")
def serve_frontend():
    return FileResponse(str(FRONTEND_DIR / "index.html"))

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

    # 学習データと同じ前処理(余白クロップ・日時スタンプ消し)をかけてからモデルに渡す。
    # これをしないと、気象庁の生のPDF変換画像(外枠・座標グリッド・日時スタンプ付き)と
    # 学習時の画像とで見た目が違いすぎて精度が大きく落ちる。
    image = autocrop_to_content(image)
    image = mask_stamp_box(image, DEFAULT_STAMP_BOX)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.sigmoid(logits)[0]

    # マルチラベル: しきい値を超えたラベルをすべて「該当する」として返す
    threshold = 0.5
    labels_above_threshold = [
        INDEX_TO_LABEL[i] for i, p in enumerate(probs) if p.item() > threshold
    ]
    return {
        "labels": labels_above_threshold,
        "labels_ja": [LABEL_JA[l] for l in labels_above_threshold],
        "all_probabilities": {
            INDEX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)
        },
        "all_probabilities_ja": {
            LABEL_JA[INDEX_TO_LABEL[i]]: float(p) for i, p in enumerate(probs)
        },
    }
