# 🌾 FarmGuardian

**AI-powered weed and crop detection with a field-severity dashboard.**
A YOLOv8 model finds crops and weeds in a field photo, and a FastAPI backend turns the detections into counts, weed density, a severity grade, a weed-concentration map and management recommendations.

![FarmGuardian dashboard](docs/dashboard.png)

## What it does

- Detects **crops** and **weeds** in a photo and draws bounding boxes with confidence scores
- Counts both, then calculates weed density and a **Low / Moderate / High** severity grade, with a plain-language explanation
- Splits the image into a grid to show **where** weeds are concentrated, so treatment can be targeted instead of applied across the whole field
- Gives management recommendations (monitor, remove locally, re-scan). It **never recommends a specific chemical** from an image alone. For high severity it tells the user to confirm with an agronomist and follow the product label, because crop type, weed species and growth stage can't be judged from a photo.
- Saves scans to a history page and compares two scans (before/after)
- Runs in a **simulated mode** when no backend is connected, so the dashboard can be demoed offline

## Results

Trained on a small public dataset, so treat these as a prototype baseline, not production accuracy.

| Metric | Value |
|---|---|
| Model | YOLOv8n (fine-tuned from pretrained weights) |
| Training / validation images | 142 / 25 |
| mAP50 | 0.716 |
| mAP50-95 | 0.447 |
| Precision | 0.742 |
| Recall | 0.624 |
| mAP50 by class | crop 0.697, weed 0.734 |

Training: 60 epochs, image size 640, batch 16, on a free Colab T4 GPU (about 3 minutes).

## Quick start

```bash
# 1. Clone the repository
git clone https://github.com/Sakshi-hugar/FarmGuardian.git
cd FarmGuardian

# 2. Put your trained weights here, named exactly this:
#    farmguardian_best.pt  

# 3. Install dependencies and run
pip install -r requirements.txt
uvicorn main:app --port 8000
```


Open **http://localhost:8000** for the dashboard. The page detects the backend at the same address and uses the real model automatically. Open **http://localhost:8000/docs** to try the API directly with Swagger UI.

### Docker

```bash
docker build -t farmguardian .
docker run -p 8000:8000 farmguardian
```

## API

### `POST /analyze`

Multipart form upload. Fields: `image` (required) and `conf` (optional confidence threshold, default 0.25).

```json
{
  "field_name": "photo.jpg",
  "image_width": 640,
  "image_height": 544,
  "weed_count": 1,
  "crop_count": 3,
  "density": 25.0,
  "severity": "Moderate",
  "severity_key": "moderate",
  "boxes": [
    {"type": "weed", "x": 120.5, "y": 80.2, "w": 45.0, "h": 60.1, "conf": 0.56}
  ],
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45
}
```

### `GET /health`

Confirms the model loaded and lists its classes. Check this first if `/analyze` returns a 503.

### Configuration (environment variables)

| Variable | Default | Purpose |
|---|---|---|
| `FARMGUARDIAN_MODEL_PATH` | `farmguardian_best.pt` | Path to the trained weights |
| `FARMGUARDIAN_CONF` | `0.25` | Default confidence threshold |
| `FARMGUARDIAN_IOU` | `0.45` | NMS IoU threshold |
| `FARMGUARDIAN_CONTAINMENT` | `0.4` | Overlap threshold for the duplicate-box filter below |

## How it works

```
Photo → FastAPI /analyze → YOLOv8 detector → duplicate-box filter
      → density + severity → recommendations → dashboard + history
```

**Duplicate-box filter.** Standard IoU-based NMS left several overlapping boxes on a single leaf, because a small box inside a large one has a very low IoU. After NMS, the backend also suppresses any lower-confidence box whose intersection with another box covers more than 40% of the *smaller* box's area.

**Severity.** `density = weeds / (weeds + crops)`. Below 15% is Low, 15–35% is Moderate, above 35% is High. These cut-offs are simple heuristics, not agronomically validated thresholds, and density is count-based rather than area-based.

## Limitations

- **Small dataset.** 142 training images and a 25-image validation set, so the metrics are noisy and optimistic.
- **Crop-specific.** The model learned one broadleaf crop. On a wheat photo it labelled the grass as weed, because it has never seen a grass-type crop. Weed/crop models generally need training data for each crop type.
- **Framing.** It works best on close-up, top-down photos like the training data. Wide or aerial shots degrade quickly.
- **Class imbalance in practice.** Low-confidence guesses are common. The confidence threshold is adjustable in the dashboard for that reason.
- The autonomous weed-removal machine in the architecture is a **design concept**, not an implemented feature.

## Roadmap

- Collect larger, more varied training data, with extra weed-focused examples
- Train per-crop models and evaluate on photos from different fields
- Host a persistent public demo
- Add logging and monitoring around the API
- Prototype the camera → detect → locate → remove loop in simulation

## Tech stack

Python, Ultralytics YOLOv8, FastAPI, OpenCV/PIL, vanilla HTML/JS with canvas, Docker, Roboflow (dataset), Google Colab (training).

## Dataset and credits

Trained on [weed-and-crop-detection](https://universe.roboflow.com/savaliya-smit/weed-and-crop-detection-ikixc) by Savaliya Smit on Roboflow Universe (MIT license). The dataset shipped with only a training split, so 15% of it was held out for validation.

## License

MIT. See `LICENSE`.
