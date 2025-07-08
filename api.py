from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import xgboost as xgb
import pandas as pd
from typing import List

# Carrega o modelo treinado (model.json deve estar em IA/)
model = xgb.Booster()
model.load_model("model.bin")

app = FastAPI(title="API de Predição XGBoost")

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
