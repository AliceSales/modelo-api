from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import re

app = FastAPI(title="API Modelo Legibilidade ISC")

artefato = joblib.load("modelo_completo_isc.pkl")
modelo = artefato["modelo_regressao"]
features_esperadas = artefato["features_esperadas"]
descricao = artefato["descricao"]


class CodeRequest(BaseModel):
    filename: str
    content: str


def extrair_features_basicas(code: str):
    # P_parametros: estimativa de quantidade de parâmetros em métodos
    assinaturas = re.findall(r"\(([^)]*)\)", code)
    p_parametros = 0

    for assinatura in assinaturas:
        assinatura = assinatura.strip()
        if assinatura:
            p_parametros += len([p for p in assinatura.split(",") if p.strip()])

    # C_complexos: estruturas de controle
    c_complexos = (
        code.count("if")
        + code.count("for")
        + code.count("while")
        + code.count("switch")
        + code.count("catch")
    )

    # B_booleanos: operadores booleanos
    b_booleanos = code.count("&&") + code.count("||") + code.count("!")

    # FD_desordem: proxy simples de desordem por TODO, println e linhas grandes
    fd_desordem = (
        code.count("TODO")
        + code.count("System.out.println")
        + sum(1 for line in code.splitlines() if len(line) > 100)
    )

    # PN_nomes_curtos: nomes muito curtos de variáveis
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

    pontuacao_sobrecarga = (5.0 - nota_1_a_5) * 5.0

    if pontuacao_sobrecarga <= 7.0:
        status = "Boa (Baixa Sobrecarga)"
    elif pontuacao_sobrecarga <= 15.0:
        status = "Média Sobrecarga"
    else:
        status = "Alta Sobrecarga"

    score_legibilidade_0_a_20 = 20 - pontuacao_sobrecarga

    return {
        "score": round(score_legibilidade_0_a_20, 2),
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
    features = extrair_features_basicas(data.content)
    resultado = classificar_codigo_isc(features)

    return {
        "filename": data.filename,
        "score": resultado["score"],
        "label": resultado["classificacao"],
        "warnings": [],
        "features": features,
        "pontuacao_sobrecarga": resultado["pontuacao_sobrecarga"],
        "nota_modelo_1_a_5": resultado["nota_modelo_1_a_5"]
    }