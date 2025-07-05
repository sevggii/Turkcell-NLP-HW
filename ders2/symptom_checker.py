SYMPTOM_DEFICIENCY_MAP = {
    "yorgunluk": ["B12 vitamini", "D vitamini", "Demir"],
    "saç dökülmesi": ["Çinko", "B7 vitamini (Biotin)", "Demir"],
    "kas krampları": ["Magnezyum", "Kalsiyum", "D vitamini"],
    "tırnak kırılması": ["Demir", "Çinko"],
    "baş dönmesi": ["Demir", "B12 vitamini"],
    "kemik ağrısı": ["D vitamini", "Kalsiyum"],
    # ... daha fazla semptom eklenebilir
}

def analyze_symptoms(symptom_text):
    found = set()
    for symptom, deficiencies in SYMPTOM_DEFICIENCY_MAP.items():
        if symptom in symptom_text.lower():
            found.update(deficiencies)
    return list(found) 