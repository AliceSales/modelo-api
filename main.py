from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional, Dict
import joblib
import numpy as np

app = FastAPI(title="API Modelo Legibilidade ISC")

artefato = joblib.load("modelo_completo_isc.pkl")
modelo = artefato["modelo_regressao"]
features_esperadas = artefato.get("features_esperadas", [])
descricao = artefato.get("descricao", "Modelo ISC")


class CodeRequest(BaseModel):
    filename: str
    features: Dict[str, float]


def classificar_codigo_isc(features: dict):
    x_novo = np.array([[
        features["P_parametros"],
        features["C_complexos"],
        features["B_booleanos"],
        features["FD_desordem"],
        features["PN_nomes_curtos"]
    ]])

    nota_1_a_5 = max(1.0, min(5.0, float(modelo.predict(x_novo)[0])))

    pontuacao_sobrecarga = (5.0 - nota_1_a_5) * 5.0
    score_0_a_10 = nota_1_a_5 * 2

    if pontuacao_sobrecarga <= 7.0:
        status = "Boa (Baixa Sobrecarga)"
    elif pontuacao_sobrecarga <= 15.0:
        status = "Média Sobrecarga"
    else:
        status = "Alta Sobrecarga"

    return {
        "score": round(score_0_a_10, 2),
        "pontuacao_sobrecarga": round(pontuacao_sobrecarga, 2),
        "classificacao": status,
        "nota_modelo_1_a_5": round(nota_1_a_5, 2)
    }


@app.get("/")
def health_check():
    return {
        "status": "ok",
        "message": "API do modelo ISC está online",
        "descricao": descricao,
        "features_esperadas": features_esperadas
    }


@app.post("/analyze")
def analyze_code(data: CodeRequest):
    resultado = classificar_codigo_isc(data.features)

    return {
        "filename": data.filename,
        "score": resultado["score"],
        "label": resultado["classificacao"],
        "warnings": [],
        "features": data.features,
        "pontuacao_sobrecarga": resultado["pontuacao_sobrecarga"],
        "nota_modelo_1_a_5": resultado["nota_modelo_1_a_5"]
    }