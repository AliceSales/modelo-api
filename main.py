from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="API Modelo Legibilidade")


class CodeRequest(BaseModel):
    filename: str
    content: str


@app.get("/")
def health_check():
    return {"status": "ok", "message": "API do modelo está online"}


@app.post("/analyze")
def analyze_code(data: CodeRequest):
    code = data.content

    score = 10
    warnings = []

    if "System.out.println" in code:
        score -= 2
        warnings.append("Uso de System.out.println detectado")

    if "TODO" in code:
        score -= 1
        warnings.append("Comentário TODO encontrado")

    if len(code.splitlines()) > 80:
        score -= 2
        warnings.append("Arquivo muito longo")

    if "public static void main" in code:
        score -= 1
        warnings.append("Método main detectado")

    score = max(score, 0)

    label = "Legível" if score >= 7 else "Precisa melhorar"

    return {
        "filename": data.filename,
        "score": score,
        "label": label,
        "warnings": warnings
    }