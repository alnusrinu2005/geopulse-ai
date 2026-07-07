# GeoPulse AI — Hackathon Demo

Quick-start MVP for **Build with AI: Code for Communities — Kisan Alert track**.

## Run it (takes ~2 minutes)

### 1. Backend
```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```
Backend runs at http://localhost:8000 — visit http://localhost:8000/docs for interactive API docs (great to show judges).

### 2. Frontend
Just open `frontend/index.html` directly in your browser (double-click it, or right-click → Open with browser). It calls the backend at `http://localhost:8000`.

## What it demonstrates
- **Crop recommendation engine** — `/api/v1/advisory`: estimates groundwater depth, soil profile, and recommends crops for a given lat/lon.
- **Dry-spell alerts** — `/api/v1/dry-spell-alert`: flags dry-spell risk and estimated days to next rain.
- **Crop health logging** — `/api/v1/crop-health`: simulates photo/voice-based AI diagnosis and routes to a Rythu Seva Kendra when needed.
- **Crowdsourced borewell logging** — `/api/v1/borewell-log`: the underlying "Digital Underground Twin" data pipeline — every logged borewell would improve future estimates.

The current model is **rule-based and deterministic per location** (same input → same output), which is intentional for a hackathon demo: transparent, fast, and it clearly shows the reasoning judges can follow. The `SUBMISSION.md` roadmap explains how this upgrades to a trained model (XGBoost/LightGBM) using CGWB groundwater records + Agmarknet + satellite rainfall data.

## Next if you have more time before the deadline
1. Swap hardcoded lat/lon inputs for a simple map picker (Leaflet.js is a fast add).
2. Wire `borewell-log` submissions into `advisory` so recommendations actually shift as records are added — a strong live demo moment ("watch the estimate improve as we add data").
3. Record a 2–3 min demo video: problem story → live walkthrough of all 3 flows → impact framing (see `SUBMISSION.md`).
