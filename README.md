# Predictive OS

Predictive OS is an enterprise-style predictive-maintenance decision layer for industrial sensor telemetry.

## UVP
**Predict → Gate → Audit.** Predictive OS benchmarks Random Forest, XGBoost, and LightGBM, selects a validated candidate, generates confidence-aware decisions, and routes uncertain predictions to human review.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Demo flow
1. Upload training telemetry CSV.
2. Select target and task type.
3. Train and benchmark the three models.
4. Review model score and feature impact.
5. Upload unseen telemetry in Live Inference.
6. Inspect confidence-gated decisions and export CSV.
7. Export the governance report.
