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
from src import calibration as calib
from src.labels import INDEX_TO_LABEL, LABEL_JA
from src.model import load_model as load_weights
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

model = None
# 前処理は重みに記録された入力サイズに合わせる。重みを読むまで確定しないため、
# 起動時のload_model()で差し替える(未学習時のフォールバックとして既定値を入れておく)。
transform = get_transforms(train=False)
# 確信度の校正。重みの隣に <重み名>.calib.json があれば起動時に読み込む。
# 無ければ「校正なし」として動く(生の出力をそのまま返す従来の挙動)。
calibration = calib.Calibration.identity()
# 校正ファイルが今の重みと食い違っていた場合の理由。起動時に埋まる。
calibration_error = None


@app.on_event("startup")
def load_model():
    global model, transform, calibration
    if not os.path.exists(WEIGHTS_PATH):
        # モデル未学習の段階でもAPIサーバー自体は起動できるようにしておく
        print(f"warning: weights not found at {WEIGHTS_PATH}. /predict will fail until trained.")
        return
    m, meta = load_weights(WEIGHTS_PATH, map_location=device)
    m.eval()
    m.to(device)
    transform = get_transforms(train=False, image_size=meta["image_size"])

    # 校正ファイルが別の重み用のものだった場合、未校正の値を黙って返すより
    # 配信しない方が安全なので、モデルを読み込まないまま起動する
    # (/health と /predict が理由を返す)。
    try:
        calibration = calib.load_for_weights(WEIGHTS_PATH)
    except calib.StaleCalibrationError as e:
        global calibration_error
        calibration_error = str(e)
        print(f"error: {e}")
        return

    model = m


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "calibrated": calibration.is_fitted,
        "calibration_error": calibration_error,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=calibration_error
            or "Model is not loaded yet. Train and place weights first.",
        )

    content = await file.read()
    image = Image.open(io.BytesIO(content)).convert("RGB")

    # 学習データと同じ前処理(余白クロップ・日時スタンプ消し)をかけてからモデルに渡す。
    # これをしないと、気象庁の生のPDF変換画像(外枠・座標グリッド・日時スタンプ付き)と
    # 学習時の画像とで見た目が違いすぎて精度が大きく落ちる。
    image = autocrop_to_content(image)
    image = mask_stamp_box(image, DEFAULT_STAMP_BOX)

    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)[0].cpu().numpy()

    # 生のsigmoid出力は学習時のpos_weightのぶん系統的に高く出るため、そのまま
    # 確信度として見せると「明らかに違うのに60%」が起きる。校正を通してから返す
    # (校正ファイルが無ければ素通し。詳細は src/calibration.py)。
    probs = calibration.probabilities(logits)

    # マルチラベル: しきい値を超えたラベルをすべて「該当する」として返す。
    # しきい値は校正時にラベルごとにF1が最大になる値を選んである。
    labels_above_threshold = calibration.predicted_labels(probs)
    ranking = sorted(
        ((INDEX_TO_LABEL[i], float(p)) for i, p in enumerate(probs)),
        key=lambda x: x[1],
        reverse=True,
    )
    top_label, top_prob = ranking[0]

    return {
        "labels": labels_above_threshold,
        "labels_ja": [LABEL_JA[l] for l in labels_above_threshold],
        # 校正済みかどうかで画面の注意書きを変えられるようにする
        "calibrated": calibration.is_fitted,
        # しきい値を1つも超えなかった場合、無理に1位を答えにせず「判定保留」として扱う。
        # 確信度が低いまま自信ありげに1つ選ぶのが、人が見て明らかに違う判定の出どころ。
        "top": {
            "label": top_label,
            "label_ja": LABEL_JA[top_label],
            "probability": top_prob,
            "threshold": calibration[top_label].threshold,
            "confident": bool(labels_above_threshold),
        },
        "thresholds": {label: calibration[label].threshold for label in INDEX_TO_LABEL.values()},
        "thresholds_ja": {
            LABEL_JA[label]: calibration[label].threshold for label in INDEX_TO_LABEL.values()
        },
        "all_probabilities": {
            INDEX_TO_LABEL[i]: float(p) for i, p in enumerate(probs)
        },
        "all_probabilities_ja": {
            LABEL_JA[INDEX_TO_LABEL[i]]: float(p) for i, p in enumerate(probs)
        },
    }
