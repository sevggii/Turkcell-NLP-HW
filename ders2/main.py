from fastapi import FastAPI
from pydantic import BaseModel
from symptom_checker import analyze_symptoms

app = FastAPI()

class SymptomInput(BaseModel):
    symptoms: str

@app.post("/analyze_symptoms")
def analyze(symptom_input: SymptomInput):
    deficiencies = analyze_symptoms(symptom_input.symptoms)
    if deficiencies:
        return {
            "possible_deficiencies": deficiencies,
            "recommendation": "Kan tahlili yaptırmanız önerilir."
        }
    else:
        return {
            "possible_deficiencies": [],
            "recommendation": "Belirtileriniz için bir uzmana danışınız."
        } 