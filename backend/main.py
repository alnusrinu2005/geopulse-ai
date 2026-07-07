"""
GeoPulse AI - Demo Backend (Hackathon MVP, v2)
Rule-based groundwater + crop recommendation engine, now with:
  1. Gemini integration (Google AI) for farmer-friendly advisory text in
     English / Telugu / Hindi, and for crop-health note analysis.
     Set GEMINI_API_KEY env var to enable; falls back to rule-based text if absent.
  2. Crowdsourced borewell records now FEED the advisory: logs within ~5 km
     of a query blend into the groundwater estimate ("Digital Underground Twin"
     demonstrated live: log a record, re-run advisory, watch the estimate shift).
"""

import hashlib
import math
import os
from datetime import datetime
from typing import Optional

import requests as http_requests
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
# Gemini (Google AI) integration
# ---------------------------------------------------------------------------
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)

LANGUAGE_NAMES = {"en": "English", "te": "Telugu", "hi": "Hindi"}


def gemini_generate(prompt: str) -> Optional[str]:
    """Call Gemini. Returns None if no key or on any error (caller falls back)."""
    if not GEMINI_API_KEY:
        return None
    try:
        resp = http_requests.post(
            GEMINI_URL,
            params={"key": GEMINI_API_KEY},
            json={"contents": [{"parts": [{"text": prompt}]}]},
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    except Exception:
        return None


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


def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Haversine distance in km."""
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _nearby_records(lat: float, lon: float, radius_km: float = 5.0) -> list[dict]:
    return [
        rec for rec in BOREWELL_LOG
        if _distance_km(lat, lon, rec["latitude"], rec["longitude"]) <= radius_km
    ]


def estimate_groundwater(lat: float, lon: float) -> dict:
    """Baseline rule-based estimate, BLENDED with nearby crowdsourced records.

    Blending: distance-weighted average of nearby logged water-strike depths,
    weighted against the baseline by record count (more records -> more trust
    in crowdsourced data). This is the live 'Digital Underground Twin' moment.
    """
    r = _seeded_random(lat, lon, "water")
    baseline_depth = 80 + r * 350  # 80 - 430 ft range
    soil = SOIL_TYPES[int(_seeded_random(lat, lon, "soil") * len(SOIL_TYPES))]

    nearby = _nearby_records(lat, lon)
    strikes = [rec for rec in nearby if rec.get("water_strike_depth_ft")]

    if strikes:
        # Inverse-distance weighting of real records
        weighted_sum, weight_total = 0.0, 0.0
        for rec in strikes:
            d = max(_distance_km(lat, lon, rec["latitude"], rec["longitude"]), 0.1)
            w = 1.0 / d
            weighted_sum += rec["water_strike_depth_ft"] * w
            weight_total += w
        crowd_depth = weighted_sum / weight_total
        # Trust crowdsourced data more as record count grows (caps at 85%)
        crowd_weight = min(0.85, 0.4 + 0.15 * len(strikes))
        depth_ft = round(crowd_weight * crowd_depth + (1 - crowd_weight) * baseline_depth)
        successes = sum(1 for rec in nearby if rec.get("water_strike_depth_ft"))
        nearby_success_rate = round(100 * successes / len(nearby))
        data_source = f"Blended: {len(strikes)} nearby borewell record(s) + baseline model"
    else:
        depth_ft = round(baseline_depth)
        nearby_success_rate = round(40 + _seeded_random(lat, lon, "success") * 55)
        data_source = "Baseline model (no nearby crowdsourced records yet)"

    probability = round(max(0.15, 1 - (depth_ft / 500)), 2)
    recharge_potential = "High" if r > 0.66 else "Medium" if r > 0.33 else "Low"
    return {
        "estimated_depth_ft": depth_ft,
        "groundwater_probability": probability,
        "soil_profile": soil,
        "nearby_borewell_success_rate_pct": nearby_success_rate,
        "recharge_potential": recharge_potential,
        "nearby_records_used": len(strikes),
        "data_source": data_source,
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
    language: str = "en"  # en | te | hi — used for Gemini farmer summary


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
    language: str = "en"
    # photo_base64: Optional[str] = None  # add when wiring real image upload


# ---------------------------------------------------------------------------
# Endpoints — mapped 1:1 to the Kisan Alert challenge's 3 components
# ---------------------------------------------------------------------------
@app.get("/")
def root():
    return {
        "status": "ok",
        "service": "GeoPulse AI demo API",
        "gemini_enabled": bool(GEMINI_API_KEY),
    }


@app.post("/api/v1/advisory")
def get_advisory(req: LocationRequest):
    """Component 1: Smart crop recommendation engine (Gemini-powered summary)."""
    gw = estimate_groundwater(req.latitude, req.longitude)
    crops = recommend_crops(gw["soil_profile"], gw["groundwater_probability"])

    lang = LANGUAGE_NAMES.get(req.language, "English")
    farmer_summary = gemini_generate(
        f"You are an agricultural advisor for Indian farmers. Based on this data, "
        f"write a short, warm, practical advisory (4-5 sentences, plain words, no jargon) "
        f"in {lang} for a farmer at this location"
        f"{' in ' + req.village_name if req.village_name else ''}:\n"
        f"- Estimated groundwater depth: {gw['estimated_depth_ft']} ft\n"
        f"- Groundwater probability: {gw['groundwater_probability']}\n"
        f"- Soil: {gw['soil_profile']}\n"
        f"- Nearby borewell success rate: {gw['nearby_borewell_success_rate_pct']}%\n"
        f"- Recommended crops: {', '.join(crops['recommended_crops'])}\n"
        f"- Irrigation note: {crops['irrigation_advice']}\n"
        f"Cover: whether drilling here is advisable, what to plant, and one water-saving tip. "
        f"Respond with only the advisory text."
    )

    return {
        "location": {"lat": req.latitude, "lon": req.longitude, "village": req.village_name},
        "groundwater": gw,
        **crops,
        "farmer_summary": farmer_summary,          # Gemini text (or null if key not set)
        "summary_source": "gemini" if farmer_summary else "rule-based",
    }


@app.post("/api/v1/dry-spell-alert")
def get_dry_spell_alert(req: LocationRequest):
    """Component 2: Real-time advisory & dry-spell alerts."""
    return dry_spell_alert(req.latitude, req.longitude)


@app.post("/api/v1/crop-health")
def log_crop_health(req: CropHealthRequest):
    """Component 3: Crop health logging -> AI diagnosis -> Rythu Seva Kendra routing.

    If the farmer typed a note and Gemini is enabled, Gemini analyses the note.
    Otherwise falls back to the deterministic demo diagnosis.
    """
    diagnosis, confidence, ai_source = None, None, "rule-based"

    if req.farmer_note:
        lang = LANGUAGE_NAMES.get(req.language, "English")
        gemini_reply = gemini_generate(
            f"You are a crop-health assistant for Indian farmers. A farmer reports: "
            f"\"{req.farmer_note}\". In {lang}, give a one-line likely diagnosis and one "
            f"immediate action step (2 sentences max). If it sounds serious, say an expert "
            f"visit is recommended. Respond with only the diagnosis text."
        )
        if gemini_reply:
            diagnosis, confidence, ai_source = gemini_reply, 0.8, "gemini"

    if diagnosis is None:
        r = _seeded_random(req.latitude, req.longitude, "health")
        conditions = ["Leaf blight (early stage)", "Nutrient deficiency", "Healthy — no action needed", "Pest infestation (aphids)"]
        diagnosis = conditions[int(r * len(conditions))]
        confidence = round(0.6 + r * 0.35, 2)

    needs_expert = "healthy" not in diagnosis.lower()
    return {
        "ai_preliminary_diagnosis": diagnosis,
        "confidence": confidence,
        "ai_source": ai_source,
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
