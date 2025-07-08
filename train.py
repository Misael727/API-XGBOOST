import pandas as pd
import xgboost as xgb

# Exemplo: use seu CSV real em vez de dados fake
df = pd.DataFrame({
    "idade": [45,30,60,15,70],
    "sexo": [1,2,1,2,1],
    "classificacao_risco": [2,3,1,4,5],
    "tempo_espera": [15,30,5,20,45],
    "qtd_exames": [2,1,3,0,2],
    "y": [1,0,1,0,1]
})

X = df[["idade","sexo","classificacao_risco","tempo_espera","qtd_exames"]]
y = df["y"]

model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
model.fit(X, y)
model.get_booster().save_model("model.bin")
print("✅  Modelo salvo em model.json")
