"""
FarmGuardian backend — serves real weed/crop detections from a trained YOLOv8 model.

Setup:
    1. Put your trained weights file next to this script, named `farmguardian_best.pt`
       (the file you downloaded from the Colab training notebook).
    2. pip install -r requirements.txt
    3. uvicorn main:app --reload --port 8000
    4. Open http://localhost:8000/docs to test it directly (upload an image, see the
       JSON response) — that alone is a legitimate live demo of the real model.
    5. Optionally, open farmguardian.html in your browser and paste
       http://localhost:8000 into the "Backend URL" field on the Analyze page to see
       real detections drawn in the dashboard instead of simulated ones.
"""

import io
import os

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from ultralytics import YOLO

MODEL_PATH = os.environ.get("FARMGUARDIAN_MODEL_PATH", "farmguardian_best.pt")
CONF_THRESHOLD = float(os.environ.get("FARMGUARDIAN_CONF", "0.25"))
IOU_THRESHOLD = float(os.environ.get("FARMGUARDIAN_IOU", "0.45"))
CONTAINMENT_THRESHOLD = float(os.environ.get("FARMGUARDIAN_CONTAINMENT", "0.4"))

app = FastAPI(title="FarmGuardian Inference API", version="1.0")

# CORS wide open for local hackathon/demo use. If you deploy this beyond your own
# machine, replace allow_origins=["*"] with your actual frontend's URL.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None


@app.on_event("startup")
def load_model():
    global model
    if not os.path.exists(MODEL_PATH):
        print(
            f"WARNING: model file '{MODEL_PATH}' not found. "
            f"Place your trained farmguardian_best.pt next to main.py, "
            f"or set FARMGUARDIAN_MODEL_PATH."
        )
        return
    model = YOLO(MODEL_PATH)
    print(f"Loaded model from {MODEL_PATH}. Classes: {model.names}")


@app.get("/health")
def health():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "model_path": MODEL_PATH,
        "classes": model.names if model is not None else None,
    }


def compute_severity(weed_count: int, crop_count: int):
    total = weed_count + crop_count
    density = round((weed_count / total) * 100, 1) if total > 0 else 0.0
    if density < 15:
        severity, sev_key = "Low", "low"
    elif density < 35:
        severity, sev_key = "Moderate", "moderate"
    else:
        severity, sev_key = "High", "high"
    return density, severity, sev_key


def suppress_contained_boxes(boxes, containment_threshold=0.6):
    """
    Standard IoU-based NMS misses a common pattern here: a small box sitting mostly
    *inside* a much larger box has low IoU (since IoU divides by the combined area),
    so it survives normal NMS even though it's clearly a duplicate/spurious detection
    of part of the same plant. This instead measures overlap relative to the SMALLER
    box's own area ("intersection over minimum"), which catches containment regardless
    of the size difference between the two boxes, and keeps the higher-confidence one.
    """
    def area(b):
        return max(0.0, b["w"]) * max(0.0, b["h"])

    def intersection(a, b):
        ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"]+a["w"], a["y"]+a["h"]
        bx1, by1, bx2, by2 = b["x"], b["y"], b["x"]+b["w"], b["y"]+b["h"]
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        return (ix2 - ix1) * (iy2 - iy1)

    ordered = sorted(boxes, key=lambda b: b["conf"], reverse=True)
    kept = []
    for b in ordered:
        b_area = area(b)
        suppressed = False
        for k in kept:
            inter = intersection(b, k)
            min_area = min(b_area, area(k))
            if min_area > 0 and (inter / min_area) > containment_threshold:
                suppressed = True
                break
        if not suppressed:
            kept.append(b)
    return kept


@app.post("/analyze")
async def analyze(image: UploadFile = File(...), conf: float = Form(None)):
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"Model not loaded — place farmguardian_best.pt next to main.py "
                   f"(looked for it at '{MODEL_PATH}') and restart the server.",
        )

    contents = await image.read()
    try:
        pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read image: {e}")

    width, height = pil_image.size

    effective_conf = conf if conf is not None else CONF_THRESHOLD

    results = model.predict(
        source=pil_image,
        conf=effective_conf,
        iou=IOU_THRESHOLD,
        agnostic_nms=True,  # suppress overlapping boxes across classes, not just within one class
        verbose=False,
    )
    result = results[0]

    boxes = []

    for box in result.boxes:
        cls_id = int(box.cls)
        cls_name = model.names[cls_id]
        conf = float(box.conf)
        x1, y1, x2, y2 = [float(v) for v in box.xyxy[0]]

        boxes.append({
            "type": cls_name,
            "x": round(x1, 1),
            "y": round(y1, 1),
            "w": round(x2 - x1, 1),
            "h": round(y2 - y1, 1),
            "conf": round(conf, 3),
        })

    boxes = suppress_contained_boxes(boxes, containment_threshold=CONTAINMENT_THRESHOLD)
    weed_count = sum(1 for b in boxes if b["type"] == "weed")
    crop_count = sum(1 for b in boxes if b["type"] == "crop")

    density, severity, sev_key = compute_severity(weed_count, crop_count)

    return {
        "field_name": image.filename,
        "image_width": width,
        "image_height": height,
        "weed_count": weed_count,
        "crop_count": crop_count,
        "density": density,
        "severity": severity,
        "severity_key": sev_key,
        "boxes": boxes,
        "confidence_threshold": effective_conf,
        "iou_threshold": IOU_THRESHOLD,
    }


@app.get("/")
def root():
    return {
        "message": "FarmGuardian inference API is running.",
        "docs": "/docs",
        "health": "/health",
        "analyze_endpoint": "POST /analyze (multipart form field: image)",
    }