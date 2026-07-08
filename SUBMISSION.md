# Hack2skill Submission — Build with AI: Code for Communities

## Project Title
GeoPulse AI — Smart Water, Crop & Advisory Platform

## Track
🌾 Kisan Alert — Smart Water, Crop & Advisory System

## Problem Statement
Farmers routinely make two high-stakes decisions with almost no reliable data: where to drill a borewell, and what to plant. Neighbors' guesses ("180 feet," "250 feet," "no water even at 400 feet") substitute for real information. A single failed borewell can cost a family ₹2–3 lakh and push them into debt, and crop choices made without soil, groundwater, or rainfall data lead to further losses. Meanwhile, every borewell ever drilled generates valuable data — GPS location, depth, soil layers, water strike depth, yield — that is thrown away the moment drilling ends. Governments have no live groundwater map, and there is no continuously updated underground picture of the land.

## Our Solution
GeoPulse AI turns every borewell into a data point and every data point into a decision-support tool for the next farmer — delivered as a voice-and-SMS-first advisory platform in Indic languages, built directly around the three asks of the Kisan Alert brief:

**1. Smart crop recommendation engine**
Combines crowdsourced borewell records (depth, soil layers, water strike, yield) with satellite imagery, rainfall history, and terrain data to estimate groundwater depth and soil profile at any location, then recommends suitable crops for that specific plot — not generic regional advice.

**2. Real-time advisory & dry-spell alerts**
Uses localized weather forecasts and the same underlying groundwater/soil model to send irrigation and fertilization alerts by SMS/voice, timed to actual dry-spell risk in the farmer's micro-region rather than district-wide averages.

**3. Crop health logging for AI diagnosis**
Farmers log crop issues by photo or voice; the system provides an initial AI read and routes the case to the nearest Rythu Seva Kendra for expert follow-up, closing the loop between diagnosis and real agronomic support.

**Underneath all three:** a growing "Digital Underground Twin" — every borewell drilling operator who logs a record (location, depth, soil, water strike, yield) makes the model smarter for every farmer near them, at zero marginal data-collection cost to the platform.

## Why This Is Different
Most crop-advisory tools give regional, static advice. GeoPulse AI's recommendations are grounded in real, hyperlocal groundwater and soil evidence that improves continuously as more borewells are logged — so accuracy compounds over time instead of staying fixed.

## Tech Stack
- **Backend:** FastAPI (Python), PostgreSQL/PostGIS for geospatial data
- **AI:** Google Gemini (advisory generation in Telugu/Hindi/English + crop-photo diagnosis via Gemini Vision); rule-based groundwater scoring blended live with crowdsourced records; roadmap to XGBoost/LightGBM trained on Agmarknet + CGWB + satellite rainfall data
- **Live data:** Open-Meteo 7-day forecast for dry-spell alerts; OpenStreetMap Nominatim for place search/reverse geocoding
- **Frontend:** React Native (Expo) for the farmer app; simple web dashboard for government/analytics view
- **Alerts:** SMS/voice gateway integration (e.g., Twilio/Exotel-style) for non-smartphone reach
- **Data sources (roadmap):** Central Ground Water Board (CGWB) records, Agmarknet, satellite rainfall/terrain APIs, crowdsourced drilling logs

## Roadmap (Beyond the Hackathon)
- **Soil Health Card integration:** farmers upload their government Soil Health Card (photo/PDF); Gemini extracts NPK, pH, and micronutrient values to sharpen crop recommendations.
- **IoT sensor ingestion:** plug-in soil moisture / water level probes streaming directly into the Digital Underground Twin.
- **Per-layer soil testing reports:** structured lab reports attached to each drilled layer — enabling detection of mineral and rare-earth indicators from the same crowdsourced drilling data, opening a second (exploration) revenue stream.
- **Water quality module:** TDS logging is live in the MVP; roadmap adds full potability lab-report uploads and quality maps for drinking vs irrigation suitability.
- **Trained ML models:** replace the baseline estimator with XGBoost/LightGBM trained on CGWB + Agmarknet + satellite rainfall data.

## Impact
- **Farmers:** avoid wasted borewell spend, get crop choices grounded in real soil/water data, timely irrigation alerts
- **Government:** live groundwater stress maps, drought risk zones, data for recharge planning
- **Borewell operators:** become recognized data contributors with a trust score, turning informal expertise into a shared public resource
- **Environment:** identifies recharge pit / farm pond / check dam opportunities from the same data

## Current Status / What We Built for This Hackathon
A working MVP: a backend API that takes a location and returns an estimated groundwater depth, soil profile, and crop recommendation using a transparent rule-based model seeded with representative data, plus a simple front-end demo showing the full farmer-facing flow end-to-end.

## Team
[Your name(s) here] — solo/[N]-member team, Andhra Pradesh

## Links
- GitHub: [repo link]
- Demo video: [link]
