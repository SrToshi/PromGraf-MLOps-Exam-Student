import logging
import datetime
import io
import zipfile
from typing import Any, Optional

import pandas as pd
import requests
from sklearn.ensemble import RandomForestRegressor
from evidently import Report, Dataset, DataDefinition, Regression
from evidently.metrics import MAE, RMSE, R2Score
from evidently.presets import DataDriftPreset

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
    try:
        input_df = pd.DataFrame([article.model_dump()])[NUM_FEATS + CAT_FEATS]
        prediction = model.predict(input_df)[0]
        return PredictionOutput(predicted_count=float(prediction))
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")