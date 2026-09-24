#!/bin/bash
API_URL="http://localhost:8080/evaluate"

echo "Firing alert test: sending synthetic data designed to produce a high RMSE..."
echo "This targets the 'HighModelRMSE' alert rule defined in Grafana (threshold: model_rmse_score > 20)."

curl -s -X POST "$API_URL" -H 'Content-Type: application/json' -d '{
  "data": [
    {"temp": 0.3, "atemp": 0.31, "hum": 0.6, "windspeed": 0.15, "mnth": 2, "hr": 8, "weekday": 1, "season": 1, "holiday": 0, "workingday": 1, "weathersit": 1, "cnt": 120},
    {"temp": 0.28, "atemp": 0.29, "hum": 0.65, "windspeed": 0.1, "mnth": 2, "hr": 9, "weekday": 1, "season": 1, "holiday": 0, "workingday": 1, "weathersit": 2, "cnt": 95}
  ],
  "evaluation_period_name": "fire_alert_test"
}'