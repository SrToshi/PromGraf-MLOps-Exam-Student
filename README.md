# Bike Sharing Monitoring — Prometheus & Grafana Exam from Benjami Moreno Torres

FastAPI regression service (`RandomForestRegressor`) predicting bike-sharing
demand, instrumented end-to-end with Prometheus metrics, Evidently-based
drift/performance evaluation, and fully provisioned Grafana dashboards/alerts.

## Prerequisites
- Docker & Docker Compose
- GNU Make

## Quick start
```bash
make                # equivalent to `make all` — builds and starts everything
make traffic         # simulate /predict traffic (default 30 requests)
make evaluation       # run real evaluation on February data, update model metrics
make fire-alert       # intentionally trigger the HighModelRMSE Grafana alert
make stop            # stop all services
make help            # list all available targets
```

Services: `bike-api` (8080), `prometheus` (9090), `grafana` (3000), `node-exporter` (9100).

## Repository structure
```
├── deployment/
│   ├── prometheus/
│   │   ├── prometheus.yml          # scrapes bike_api + node_exporter, loads rules/
│   │   └── rules/
│   │       └── alert_rules.yml     # BikeApiDown alert (up == 0 for 2m)
│   └── grafana/
│       ├── dashboards/             # the 3 exported dashboard JSONs (Dashboards as Code)
│       │   ├── API-dashboard.json
│       │   ├── Model-dashboard.json
│       │   └── Infra-dashboard.json
│       └── provisioning/
│           ├── datasources/
│           │   └── datasources.yaml   # provisions Prometheus with a FIXED uid
│           └── dashboards/
│               └── dashboards.yaml    # points Grafana at ./deployment/grafana/dashboards
├── src/
│   ├── api/            # bike-api service
│   └── evaluation/     # run_evaluation.py, provided as-is by the exam
├── docker-compose.yml
├── Makefile
├── generate_predict_traffic.sh   # traffic generator required by point V
├── fire_alert.sh                 # used by `make fire-alert`
└── README.md
```

## Design decisions

### Chosen additional metric: MAPE + Data Drift Status
Beyond the required RMSE, MAE and R2Score, two extra signals are tracked:

- **`model_mape_score`**: the model's error as a percentage rather than an
  absolute count — easier to interpret at a glance than RMSE/MAE alone, since
  `cnt` has no fixed scale reference.
- **`evidently_data_drift_detected_status`** (0/1 Gauge): derived from
  Evidently's `DataDriftPreset`, specifically its `DriftedColumnsCount` metric,
  flagging when the share of drifted columns crosses the 0.5 threshold. Early
  drift detection on input features can warn of upcoming performance
  degradation before RMSE/MAE fully reflect it.

### Dashboards as Code — datasource UID pinning
`datasources.yaml` provisions the Prometheus datasource with an explicit,
fixed `uid`. The 3 exported dashboard JSONs reference that same uid directly
. Pinning the uid on the datasource side guarantees the dashboards resolve
correctly on any fresh machine running `make`, without needing to hand-edit the JSONs.

### Alerts
- **Prometheus** — `BikeApiDown` (`up{job="bike_api"} == 0` for 2m): the API
  being unreachable.
- **Grafana** — `HighModelRMSE` (`model_rmse_score > 20` for 2m, folder
  `MLOps Alerts`): an ML-performance alert, notifying via a webhook contact
  point.

### `fire-alert` target
`make fire-alert` runs `fire_alert.sh`, sending a small synthetic batch to
`/evaluate` whose `cnt` values are deliberately inconsistent with what the
January-trained reference model predicts for similar features. This forces
`model_rmse_score` well above 20, triggering `HighModelRMSE`. Run
`make evaluation` afterwards to restore realistic metrics from real February
data.

### `traffic` target (beyond the required Makefile targets)
`generate_predict_traffic.sh` satisfies point V's requirement for a script
generating `/predict` traffic. It's wired into the Makefile as `make traffic`
(optionally `make traffic N=50`) purely for convenience — not one of the four
mandatory targets, but consistent with the brief's preference for automation.
