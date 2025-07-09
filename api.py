from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
from typing import List

# Carrega o modelo treinado
model = xgb.Booster()
model.load_model("model.bin")

app = FastAPI(title="API de Predição XGBoost")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.synapseia.tech", "https://synapseia.tech"],  # ou ["*"] em dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Payload(BaseModel):
    idade: int
    sexo: int
    classificacao_risco: int
    tempo_espera: int
    qtd_exames: int

@app.post("/predict")
def predict(payloads: List[Payload]):
    df = pd.DataFrame([p.dict() for p in payloads])
    dmat = xgb.DMatrix(df)
    try:
        preds = model.predict(dmat)
        return {"predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
