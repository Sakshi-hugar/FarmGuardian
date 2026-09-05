# FarmGuardian Backend

Serves real detections from your trained YOLOv8 model (`farmguardian_best.pt`) over an HTTP API.

## Setup (5 minutes)

```bash
# 1. Put your trained weights here, named exactly this:
#    farmguardian_best.pt 

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the server
uvicorn main:app --reload --port 8000
```

## Try it immediately — no frontend needed

Open **http://localhost:8000/docs** in your browser. This gives you FastAPI's built-in
Swagger UI: expand `POST /analyze`, click "Try it out", upload a field photo, hit
Execute. You'll get back real JSON with detected boxes, counts, density, and severity —
straight from your trained model. This alone is a legitimate live demo for your
ideathon, even without wiring up the web app.

## Connect it to the FarmGuardian demo app

1. Keep this server running (`uvicorn main:app --reload --port 8000`)
2. Open `farmguardian.html` **directly in your browser** (double-click the file, or
   drag it into a browser tab) — not through a sandboxed preview, since that can block
   requests to `localhost`
3. Go to the **Analyze Field** page
4. Paste `http://localhost:8000` into the **Backend URL** field
5. Upload a photo and click Analyze — you'll now see real detections instead of
   simulated ones, drawn by the same dashboard you already built

If you leave the Backend URL field empty, the app falls back to simulated detections
automatically — so the demo still works even if the backend isn't running (useful as a
safety net during your actual presentation).

## API reference

### `POST /analyze`
Multipart form upload, field name `image`. Returns:

```json
{
  "field_name": "photo.jpg",
  "image_width": 640,
  "image_height": 544,
  "weed_count": 6,
  "crop_count": 0,
  "density": 100.0,
  "severity": "High",
  "severity_key": "high",
  "boxes": [
    {"type": "weed", "x": 120.5, "y": 80.2, "w": 45.0, "h": 60.1, "conf": 0.35}
  ],
  "confidence_threshold": 0.25
}
```

### `GET /health`
Confirms the model loaded and lists its class names — hit this first if `/analyze`
returns a 503.

## Notes

- `CONF_THRESHOLD` defaults to 0.25. Given the model was trained on only 142 images,
  you may want to lower it (e.g. `FARMGUARDIAN_CONF=0.15 uvicorn main:app --port 8000`)
  for photos that don't closely match the training set's close-up framing.
- CORS is wide open (`allow_origins=["*"]`) for easy local demo use. Tighten this before
  deploying anywhere beyond your own machine.
