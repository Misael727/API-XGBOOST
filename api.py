import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List

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
    url = "https://api.synapseia.tech/predict"
    headers = {'Content-Type': 'application/json'}

    # Converte o payload para o formato esperado pela API externa
    data = [
        {
            "idade": p.idade,
            "sexo": p.sexo,
            "classificacao_risco": p.classificacao_risco,
            "tempo_espera": p.tempo_espera,
            "qtd_exames": p.qtd_exames
        }
        for p in payloads
    ]
    
    try:
        # Envia os dados para a API de predição externa
        response = requests.post(url, json=data, headers=headers)

        # Verifica se a resposta foi bem-sucedida
        if response.status_code == 200:
            return {"predictions": response.json()["predictions"]}
        else:
            raise HTTPException(status_code=response.status_code, detail=response.text)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
