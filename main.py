from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI(title="API Modelo Legibilidade ISC")

artefato = joblib.load("modelo_completo_isc.pkl")
modelo = artefato["modelo_regressao"]
features_esperadas = artefato.get("features_esperadas", [])
descricao = artefato.get("descricao", "Modelo ISC")


class CodeRequest(BaseModel):
    filename: str
    content: str


def extrair_features_basicas(code: str):
    assinaturas = re.findall(r"\(([^)]*)\)", code)
    p_parametros = 0

    for assinatura in assinaturas:
        assinatura = assinatura.strip()
        if assinatura:
            p_parametros += len([p for p in assinatura.split(",") if p.strip()])

    c_complexos = (
        code.count("if")
        + code.count("for")
        + code.count("while")
        + code.count("switch")
        + code.count("catch")
    )

    b_booleanos = code.count("&&") + code.count("||") + code.count("!")

    fd_desordem = (
        code.count("TODO")
        + code.count("System.out.println")
        + sum(1 for line in code.splitlines() if len(line) > 100)
    )

    nomes_variaveis = re.findall(
        r"\b(?:int|double|float|String|boolean|char|long)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
        code
    )

    pn_nomes_curtos = sum(1 for nome in nomes_variaveis if len(nome) <= 2)

    return {
        "P_parametros": p_parametros,
        "C_complexos": c_complexos,
        "B_booleanos": b_booleanos,
        "FD_desordem": fd_desordem,
        "PN_nomes_curtos": pn_nomes_curtos
    }


def classificar_codigo_isc(features: dict):
    x_novo = np.array([[
        features["P_parametros"],
        features["C_complexos"],
        features["B_booleanos"],
        features["FD_desordem"],
        features["PN_nomes_curtos"]
    ]])

    nota_1_a_5 = float(modelo.predict(x_novo)[0])
    nota_1_a_5 = max(1.0, min(5.0, nota_1_a_5))

    score_0_a_10 = nota_1_a_5 * 2

    if score_0_a_10 >= 7:
        status = "Boa legibilidade"
    elif score_0_a_10 >= 5:
        status = "Legibilidade média"
    else:
        status = "Baixa legibilidade"

    return {
        "score": round(score_0_a_10, 2),
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
    features = extrair_features_basicas(data.content)
    resultado = classificar_codigo_isc(features)

    return {
        "filename": data.filename,
        "score": resultado["score"],
        "label": resultado["classificacao"],
        "warnings": [],
        "features": features,
        "nota_modelo_1_a_5": resultado["nota_modelo_1_a_5"]
    }