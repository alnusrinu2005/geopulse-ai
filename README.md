# GeoPulse AI — Hackathon Demo (v2)

MVP for **Build with AI: Code for Communities — Kisan Alert track**.

## Run it (~2 minutes)

### 1. Backend
```bash
cd backend
pip install -r requirements.txt

# Enable Gemini (recommended for the demo — this IS the "Build with AI" part):
# Get a free key at https://aistudio.google.com/apikey
set GEMINI_API_KEY=your_key_here        # Windows CMD
# $env:GEMINI_API_KEY="your_key_here"   # PowerShell

uvicorn main:app --reload --port 8000
```
Visit http://localhost:8000/docs for interactive API docs (great to show judges).
Without a key, everything still works — Gemini features fall back to rule-based text.

### 2. Frontend
Open `frontend/index.html` in your browser. Needs internet (Leaflet map tiles from CDN).

## What it demonstrates
- **Interactive map picker** — tap anywhere in India to select a plot.
- **Crop recommendation engine** (`/api/v1/advisory`) — groundwater estimate, soil profile,
  crop recommendations, **plus a Gemini-written farmer advisory in English / Telugu / Hindi**.
- **Dry-spell alerts** (`/api/v1/dry-spell-alert`).
- **Crop health check** (`/api/v1/crop-health`) — farmer describes the problem in words,
  **Gemini diagnoses it** and routes serious cases to a Rythu Seva Kendra.
- **Digital Underground Twin — LIVE** (`/api/v1/borewell-log`) — logged borewell records
  within ~5 km now blend into the groundwater estimate via inverse-distance weighting.

## The demo-video money shot (do this on camera)
1. Tap a location → run advisory → note estimated depth (e.g., 127 ft, "Baseline model").
2. Log a borewell record nearby with water strike at 210 ft.
3. Re-run advisory → estimate shifts (e.g., 181 ft), source line now reads
   **"Blended: 2 nearby borewell record(s) + baseline model"**, success rate updates.
4. Say: "Every borewell logged makes the prediction better for every farmer nearby —
   at zero data-collection cost."

## Architecture note for judges
The baseline estimator is deliberately rule-based and deterministic (transparent, judge-followable).
The blending layer shows the real data pipeline. Roadmap: replace baseline with
XGBoost/LightGBM trained on CGWB groundwater records + Agmarknet + satellite rainfall.
