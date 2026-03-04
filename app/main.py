from fastapi import FastAPI
import joblib
import numpy as np

app = FastAPI()

# Load trained model
model = joblib.load("models/car_mpg_model.pkl")

@app.get("/")
def home():
    return {"message": "Car MPG Prediction API"}

@app.get("/predict")
def predict(hp: float, vol: float, wt: float):
    input_data = np.array([[hp, vol, wt]])
    prediction = model.predict(input_data)
    return {"predicted_mpg": float(prediction[0])}
import subprocess

@app.post("/retrain")
def retrain_model():
    try:
        subprocess.run(["python", "app/train_model.py"], check=True)
        return {"message": "Model retrained successfully"}
    except Exception as e:
        return {"error": str(e)}
import psycopg2
import os

@app.get("/health")
def health_check():
    status = {}

    status["api_status"] = "running"

    if os.path.exists("car_mpg_model.pkl"):
        status["model_status"] = "loaded"
    else:
        status["model_status"] = "missing"

    try:
        conn = psycopg2.connect(
            dbname="car_project",
            user="postgres",
            password="1234",
            host="postgres"
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM cars;")
        rows = cursor.fetchone()[0]

        status["database_status"] = "connected"
        status["data_rows"] = rows

        conn.close()

    except Exception as e:
        status["database_status"] = "error"
        status["error"] = str(e)

    return status
