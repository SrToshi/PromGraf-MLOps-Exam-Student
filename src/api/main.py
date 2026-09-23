import logging
import datetime
import io
import zipfile
import time
from typing import Any, Optional

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset
from evidently.metrics import MAE, RMSE, R2Score, MAPE

from fastapi import FastAPI, HTTPException, Response, Request
from pydantic import BaseModel, Field

from prometheus_client import Counter, Histogram, generate_latest, CollectorRegistry, Gauge


# --- Logging Configuration ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- FastAPI App Initialization ---
app = FastAPI(
    title="Bike Sharing Predictor API",
    description="API for predicting bike sharing demand with MLOps monitoring.",
    version="1.0.0"
)

# --- Prometheus Metrics Definitions ---
registry = CollectorRegistry()

api_requests_total = Counter(
    'api_requests_total',
    'Total number of API requests',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

api_request_duration_seconds = Histogram(
    'api_request_duration_seconds',
    'API request duration in seconds',
    ['endpoint', 'method', 'status_code'],
    registry=registry
)

model_rmse_score = Gauge(
    'model_rmse_score',
    'RMSE of the regression model',
    registry=registry)

model_mae_score = Gauge(
    'model_mae_score',
    'MAE of the regression model',
    registry=registry
)   

model_r2_score = Gauge(
    'model_r2_score',
    'R2 Score of the regression model',
    registry=registry
)

model_mape_score = Gauge(
    'model_mape_score',
    'MAPE of the regression model',
    registry=registry
)

evidently_data_drift_detected_status = Gauge(
    'evidently_data_drift_detected_status',
    'Whether data drift was detected (1) or not (0)',
    registry=registry
)

# --- Global Variables for Model and Data ---
TARGET = 'cnt'
PREDICTION = 'prediction'
NUM_FEATS = ['temp', 'atemp', 'hum', 'windspeed', 'mnth', 'hr', 'weekday']
CAT_FEATS = ['season', 'holiday', 'workingday', 'weathersit']


# --- Data Ingestion and Preparation Functions ---
DATASET_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DTEDAY_COL_NAME = 'dteday'

def _fetch_data() -> pd.DataFrame:
    """Fetches the bike sharing dataset and returns a DataFrame."""
    logger.info("Fetching data from UCI archive...")
    try:
        content = requests.get(DATASET_URL, verify=False, timeout=60).content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            df = pd.read_csv(z.open("hour.csv"), header=0, sep=',', parse_dates=[DTEDAY_COL_NAME])
        logger.info("Data fetched successfully.")
        return df
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching data: {e}")
        raise RuntimeError("Failed to fetch dataset.") from e

def _process_data(df: pd.DataFrame, start_date: str, end_date: str) -> pd.DataFrame:
    """Filters the dataset to a date range and keeps only the columns the model needs."""
    mask = (df[DTEDAY_COL_NAME] >= start_date) & (df[DTEDAY_COL_NAME] <= end_date)
    subset = df.loc[mask, NUM_FEATS + CAT_FEATS + [TARGET]].copy()
    return subset

def _train_and_predict_reference_model(df: pd.DataFrame):
    """Trains the RandomForestRegressor on January 2011 data and returns the model plus reference predictions."""
    reference_data = _process_data(df, "2011-01-01", "2011-01-31")
    X = reference_data[NUM_FEATS + CAT_FEATS]
    y = reference_data[TARGET]

    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    reference_data[PREDICTION] = model.predict(X)
    logger.info(f"Model trained on {len(reference_data)} January 2011 records.")
    return model, reference_data

# --- Model Training at Startup ---
try:
    _raw_data = _fetch_data()
    model, reference_data = _train_and_predict_reference_model(_raw_data)
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise RuntimeError("Application cannot start without a trained model.") from e

# --- Pydantic Models for API Input/Output ---
class BikeSharingInput(BaseModel):
    temp: float = Field(..., example=0.24)
    atemp: float = Field(..., example=0.2879)
    hum: float = Field(..., example=0.81)
    windspeed: float = Field(..., example=0.0)
    mnth: int = Field(..., example=1)
    hr: int = Field(..., example=0)
    weekday: int = Field(..., example=6)
    season: int = Field(..., example=1)
    holiday: int = Field(..., example=0)
    workingday: int = Field(..., example=0)
    weathersit: int = Field(..., example=1)
    dteday: datetime.date = Field(..., example="2011-01-01", description="Date of the record in YYYY-MM-DD format.")

class PredictionOutput(BaseModel):
    predicted_count: float = Field(..., example=16.0)

class EvaluationData(BaseModel):
    data: list[dict[str, Any]] = Field(..., description="List of data points, each containing features and the true target ('cnt').")
    evaluation_period_name: str = Field("unknown_period", description="Name of the period being evaluated (e.g., 'week1_february').")
    model_config = {'arbitrary_types_allowed': True}

class EvaluationReportOutput(BaseModel):
    message: str
    rmse: Optional[float]
    mape: Optional[float]
    mae: Optional[float]
    r2score: Optional[float]
    drift_detected: int
    evaluated_items: int

# --- API Endpoints ---
@app.get("/")
async def read_root():
    return {"message": "Welcome to the Bike Sharing Predictor API. Use /predict to get bike counts or /evaluate to run drift reports."}

@app.post("/predict", response_model=PredictionOutput)
async def predict(article: BikeSharingInput):
    """Predicts the bike count for a given set of features."""
    start_time = time.time()
    status_code = "200"

    try:
        input_df = pd.DataFrame([article.model_dump()])[NUM_FEATS + CAT_FEATS]
        prediction = model.predict(input_df)[0]
        return PredictionOutput(predicted_count=float(prediction))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/predict", method="POST", status_code=status_code).inc()
        api_request_duration_seconds.labels(endpoint="/predict", method="POST", status_code=status_code).observe(duration)

@app.post("/evaluate", response_model=EvaluationReportOutput)
async def evaluate(payload: EvaluationData):
    """Runs an Evidently report comparing current data (payload) against the January reference."""
    start_time = time.time()
    status_code = "200"

    try:
        current_df = pd.DataFrame(payload.data)[NUM_FEATS + CAT_FEATS + [TARGET]].copy()
        current_df[PREDICTION] = model.predict(current_df[NUM_FEATS + CAT_FEATS])

        definition = DataDefinition(regression=[Regression(target=TARGET, prediction=PREDICTION)])
        ref_dataset = Dataset.from_pandas(reference_data, data_definition=definition)
        cur_dataset = Dataset.from_pandas(current_df, data_definition=definition)

        report = Report([RMSE(), MAE(), R2Score(), MAPE(), DataDriftPreset()])
        my_eval = report.run(current_data=cur_dataset, reference_data=ref_dataset)
        # Raw result from Evidently — already has all computed values,
        # but as a generic flat list, not indexed by metric name yet.
        result_dict = my_eval.dict()

        logger.info(f"EVIDENTLY DEBUG DICT: {result_dict}")

        # Empty placeholders — filled below, default None/0 if a metric
        # is ever missing from the result (p.ex. a future Evidently version changes the names).
        rmse_val = mae_val = r2_val = mape_val = None
        drift_share_val = 0.0

        # Translate the generic list into our named variables,
        # matching each entry by its metric_name since list order
        # isn't guaranteed once DataDriftPreset expands into one entry per column.
        for m in result_dict.get('metrics', []):
            name = m.get('metric_name', '')
            value = m.get('value')
            if name.startswith('RMSE'):
                rmse_val = float(value)
            elif name.startswith('MAE'):
                mae_val = float(value['mean']) if isinstance(value, dict) else float(value)
            elif name.startswith('R2Score'):
                r2_val = float(value)
            elif name.startswith('MAPE'):
                mape_val = float(value['mean']) if isinstance(value, dict) else float(value)
            elif name.startswith('DriftedColumnsCount'):
                drift_share_val = float(value.get('share', 0.0))

        drift_val = 1 if drift_share_val >= 0.5 else 0


        model_rmse_score.set(rmse_val or 0)
        model_mae_score.set(mae_val or 0)
        model_r2_score.set(r2_val or 0)

        model_mape_score.set(mape_val or 0)
        evidently_data_drift_detected_status.set(drift_val)

        return EvaluationReportOutput(
            message=f"Evaluation completed for period '{payload.evaluation_period_name}'",
            rmse=rmse_val,
            mape=mape_val,
            mae=mae_val,
            r2score=r2_val,
            drift_detected=drift_val,
            evaluated_items=len(current_df)
        )
    except Exception as e:
        logger.error(f"Evaluation error: {e}")
        status_code = "500"
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    finally:
        duration = time.time() - start_time
        api_requests_total.labels(endpoint="/evaluate", method="POST", status_code=status_code).inc()
        api_request_duration_seconds.labels(endpoint="/evaluate", method="POST", status_code=status_code).observe(duration)

@app.get("/metrics")
async def metrics(request: Request):
    """
    Expose Prometheus metrics.
    """
    return Response(content=generate_latest(registry), media_type="text/plain")