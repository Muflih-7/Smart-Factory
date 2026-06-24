---
title: FactoryOS
emoji: 🏭
colorFrom: gray
colorTo: yellow
sdk: docker
pinned: false
---

# FactoryOS — Unified Smart Factory Platform

**A full-stack industrial monitoring platform combining real-time factory simulation, machine learning failure prediction, and a 3D digital twin — built as a single Flask application.**

🔗 **Live demo:** [huggingface.co/spaces/Muflih-7/FactoryOS](https://huggingface.co/spaces/Muflih-7/FactoryOS)
Login: `admin` / `factory123` (full access) or `viewer` / `viewer123` (read-only)

---

## Overview

FactoryOS simulates a small fleet of industrial machines (drilling, assembly, welding stations) running in real time, then layers predictive maintenance and visualization tools on top — the kind of stack you'd find in an Industry 4.0 / smart manufacturing deployment, condensed into a single deployable app.

It started as two separate projects — a predictive maintenance system and a digital twin dashboard — and was merged into one unified platform.

## Features

**Live Factory Dashboard**
Real-time KPIs across the fleet — output, efficiency, OEE, health score — with per-machine cards showing temperature, load, and live AI risk predictions. Auto-refreshes every 3 seconds.

**3D Digital Twin**
Interactive WebGL factory floor view, hand-built without Three.js — custom shaders, particle system (weld sparks, drill chips, smoke), and click-to-select raycasting. Click any machine to inspect its live state in 3D space.

**AI Failure Prediction**
A `RandomForestClassifier` (scikit-learn, 100 estimators) trained on five sensor features — temperature, vibration, pressure, runtime hours, and oil level — predicts failure probability and estimates remaining useful life (RUL). Feature importance analysis shows **vibration and oil level are the dominant predictors** (~47% and ~36% of the model's decision weight respectively), which lines up with real-world rotating-machinery failure patterns — bearing wear and lubrication breakdown are typically the earliest and strongest failure signals, well ahead of temperature change. The dashboard also plots a 200-hour failure-probability timeline per machine.

**Scenario Simulation Engine**
Three operating modes — Normal, High Load, and Failure Simulation — drive a physics-style simulation for each machine (temperature drift, vibration buildup, efficiency loss, state transitions through Running → Warning → Failure → Maintenance).

**Analytics & Energy**
Fleet-wide OEE, MTBF, MTTR, and defect-rate tracking, plus energy consumption and carbon-emission monitoring per machine with optimization recommendations.

**Maintenance & Alerts**
AI-prioritized maintenance scheduling (critical / high / medium / low) based on live RUL and health scores, a working-order logging system, and a live + historical alert center.

**Notifications**
Live notification feed for failure predictions and maintenance events, with unread tracking.

**Reports & Export**
PDF and Excel export for factory status snapshots and full ML prediction history.

**Role-Based Access**
Admin and viewer roles; admins can rename machines, tune alert thresholds, and manage users from Settings.

**Machine Fleet Management**
Commission new machines (up to 100) or decommission existing ones on the fly.

## Tech Stack

| Layer | Technology |
|---|---|
| Backend | Flask (single-file app, server-rendered HTML) |
| ML | scikit-learn (`RandomForestClassifier`), numpy |
| Data | pandas |
| Reports | reportlab (PDF), openpyxl (Excel) |
| Auth | Werkzeug (password hashing) |
| Deployment | Gunicorn + eventlet (async workers), Docker |
| Frontend | Hand-built dark SCADA-style UI, no JS framework — inline SVG charts, custom CSS design system, hand-rolled WebGL 3D renderer |

## Architecture

- **No database** — all machine state lives in memory and resets on restart. Intentional for a live demo/portfolio context; would need persistence (Postgres/SQLite) for production use.
- **`Machine` class** — each unit tracks temperature, vibration, pressure, oil level, OEE, MTBF/MTTR, health score, and RUL, updated on a ~2-second tick driven by the active scenario mode.
- **ML inference** — on each prediction request, the five live/sensor values are fed into the pre-trained `model.pkl`; if the model fails to load, the app falls back to a heuristic health-score formula so the platform degrades gracefully rather than crashing.
- **Routes** — `/` dashboard, `/twin` 3D view, `/predict` ML interface, `/machine/<id>` detail view, `/analytics`, `/energy`, `/alerts`, `/maintenance`, `/reports`, `/settings`, `/notifications`, plus machine commission/decommission endpoints.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

Visit `http://127.0.0.1:5000` and log in with the demo credentials above.

## Roadmap

- Persistent storage for machine state and prediction history
- Configurable ML thresholds per machine type
- Multi-factory / multi-site support
- Real sensor ingestion (MQTT/OPC-UA) as an alternative to simulation mode

---

Built by [Muflih-7](https://github.com/Muflih-7) — mechanical engineering student exploring AI-driven Industry 4.0 systems.