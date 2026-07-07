"""
GeoPulse AI - Demo Backend (Hackathon MVP)
Rule-based groundwater + crop recommendation engine.
Swap the estimate_groundwater() and recommend_crops() internals with a
trained model (XGBoost/LightGBM) once real CGWB/Agmarknet data is wired in.
"""

import hashlib
import math
from datetime import datetime
from typing import Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="GeoPulse AI Demo API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# In-memory "crowdsourced borewell" store (seed data for demo)
# ---------------------------------------------------------------------------
BOREWELL_LOG: list[dict] = []

SOIL_TYPES = ["Red loamy soil", "Black cotton soil", "Sandy loam", "Alluvial soil", "Laterite soil"]

CROP_LIBRARY = {
    "Red loamy soil":  ["Groundnut", "Chillies", "Maize", "Pigeon pea"],
    "Black cotton soil": ["Cotton", "Sorghum (Jowar)", "Sunflower", "Soybean"],
    "Sandy loam":      ["Millets (Bajra)", "Watermelon", "Groundnut"],
    "Alluvial soil":   ["Paddy (Rice)", "Sugarcane", "Wheat"],
    "Laterite soil":   ["Cashew", "Tapioca", "Pineapple"],
}


def _seeded_random(lat: float, lon: float, salt: str = "") -> float:
    """Deterministic pseudo-random 0..1 value from coordinates, so demo
    results are stable for the same location instead of jumping around."""
    key = f"{round(lat, 3)}:{round(lon, 3)}:{salt}".encode()
    h = hashlib.sha256(key).hexdigest()
    return int(h[:8], 16) / 0xFFFFFFFF


def estimate_groundwater(lat: float, lon: float) -> dict:
    r = _seeded_random(lat, lon, "water")
    depth_ft = round(80 + r * 350)  # 80 - 430 ft range
    probability = round(max(0.15, 1 - (depth_ft / 500)), 2)
    soil = SOIL_TYPES[int(_seeded_random(lat, lon, "soil") * len(SOIL_TYPES))]
    nearby_success_rate = round(40 + _seeded_random(lat, lon, "success") * 55)  # 40-95%
    recharge_potential = "High" if r > 0.66 else "Medium" if r > 0.33 else "Low"
    return {
        "estimated_depth_ft": depth_ft,
        "groundwater_probability": probability,
        "soil_profile": soil,
        "nearby_borewell_success_rate_pct": nearby_success_rate,
        "recharge_potential": recharge_potential,
    }


def recommend_crops(soil: str, groundwater_probability: float) -> dict:
    crops = CROP_LIBRARY.get(soil, ["Millets (Bajra)", "Pulses"])
    if groundwater_probability < 0.35:
        irrigation_advice = "Low water availability expected — prioritize drought-resistant crops and drip irrigation."
    elif groundwater_probability < 0.65:
        irrigation_advice = "Moderate water availability — suitable for mixed cropping with scheduled irrigation."
    else:
        irrigation_advice = "Good water availability — standard irrigation cycles should be sufficient."
    return {"recommended_crops": crops, "irrigation_advice": irrigation_advice}


def dry_spell_alert(lat: float, lon: float) -> dict:
    r = _seeded_random(lat, lon, "weather")
    risk = "High" if r > 0.7 else "Moderate" if r > 0.4 else "Low"
    days_to_next_rain = int(2 + r * 12)
    return {
        "dry_spell_risk": risk,
        "estimated_days_to_next_rain": days_to_next_rain,
        "message": (
            f"Dry spell risk is {risk.lower()}. Expected rain in ~{days_to_next_rain} days. "
            "Plan irrigation accordingly." if risk != "Low"
            else "No significant dry spell expected in the near term."
        ),
    }


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------
class LocationRequest(BaseModel):
    latitude: float
    longitude: float
    village_name: Optional[str] = None


class BorewellLogRequest(BaseModel):
    latitude: float
    longitude: float
    depth_ft: float
    soil_layers: str
    water_strike_depth_ft: Optional[float] = None
    yield_lph: Optional[float] = None  # litres per hour
    operator_name: Optional[str] = None


class CropHealthRequest(BaseModel):
    latitude: float
    longitude: float
    farmer_note: Optional[str] = None
    # photo_base64: Optional[str] = None  # add when wiring real image upload


# ---------------------------------------------------------------------------
# Endpoints — mapped 1:1 to the Kisan Alert challenge's 3 components
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {"status": "ok", "service": "GeoPulse AI demo API"}


@app.post("/api/v1/advisory")
def get_advisory(req: LocationRequest):
    """Component 1: Smart crop recommendation engine."""
    gw = estimate_groundwater(req.latitude, req.longitude)
    crops = recommend_crops(gw["soil_profile"], gw["groundwater_probability"])
    return {
        "location": {"lat": req.latitude, "lon": req.longitude, "village": req.village_name},
        "groundwater": gw,
        **crops,
    }


@app.post("/api/v1/dry-spell-alert")
def get_dry_spell_alert(req: LocationRequest):
    """Component 2: Real-time advisory & dry-spell alerts."""
    return dry_spell_alert(req.latitude, req.longitude)


@app.post("/api/v1/crop-health")
def log_crop_health(req: CropHealthRequest):
    """Component 3: Crop health logging -> AI diagnosis -> Rythu Seva Kendra routing."""
    r = _seeded_random(req.latitude, req.longitude, "health")
    conditions = ["Leaf blight (early stage)", "Nutrient deficiency", "Healthy — no action needed", "Pest infestation (aphids)"]
    diagnosis = conditions[int(r * len(conditions))]
    needs_expert = diagnosis != "Healthy — no action needed"
    return {
        "ai_preliminary_diagnosis": diagnosis,
        "confidence": round(0.6 + r * 0.35, 2),
        "routed_to_rythu_seva_kendra": needs_expert,
        "nearest_kendra": "RSK - Mandal Office (demo placeholder)" if needs_expert else None,
        "logged_at": datetime.utcnow().isoformat(),
    }


@app.post("/api/v1/borewell-log")
def log_borewell(entry: BorewellLogRequest):
    """Crowdsourced borewell logging that feeds the Digital Underground Twin."""
    record = entry.dict()
    record["logged_at"] = datetime.utcnow().isoformat()
    BOREWELL_LOG.append(record)
    return {"message": "Borewell record logged", "total_records": len(BOREWELL_LOG), "record": record}


@app.get("/api/v1/borewell-log")
def list_borewell_logs():
    return {"total_records": len(BOREWELL_LOG), "records": BOREWELL_LOG}
