---
title: AirSight
emoji: 🌫️
colorFrom: blue
colorTo: gray
sdk: docker
pinned: false
app_port: 7860
---

# AirSight 🌫️

**Air Quality Estimation from Images** — ResNet-50 + RAG + Gemini 2.5 Flash

Upload an aerial or street-level image → get predicted concentrations for PM2.5, PM10, O3, CO, SO2, NO2, and AQI, plus an AI-generated health advisory grounded in official EPA/CPCB guidelines.

## API Endpoints

| Method | URL | Description |
|--------|-----|-------------|
| `GET` | `/health` | Liveness check |
| `POST` | `/predict` | CNN inference only (~2–3s) |
| `POST` | `/analyze` | CNN + FAISS + Gemini advisory (~5–10s) |
| `POST` | `/chat` | RAG-grounded Q&A |

Interactive docs: [`/docs`](https://gh0stfreak-airsight.hf.space/docs)

## Frontend

Live at: [airsight.vercel.app](https://airsight.vercel.app)

## Model Performance

Evaluated on 1,061 test images. All R² > 0.92.

| Pollutant | R² |
|---|---|
| O3 | 0.9647 |
| AQI | 0.9634 |
| CO | 0.9539 |
| PM2.5 | 0.9504 |
| PM10 | 0.9475 |
| SO2 | 0.9322 |
| NO2 | 0.9204 |
