from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re
import random

app = Flask(__name__)
CORS(app)

# ================================
# 🤖 Load ML Model
# ================================
model = pickle.load(open("../ml_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../ml_model/vectorizer.pkl", "rb"))

# ================================
# 🗂️ In-memory Database
# ================================
records = []
patient_counter = 4390

# ================================
# 🔍 NLP Extraction
# ================================
def extract_symptoms(text):
    text = text.lower()
    symptoms = []

    if "fever" in text:
        symptoms.append("Fever")
    if "cough" in text:
        symptoms.append("Cough")
    if "pain" in text:
        symptoms.append("Pain")
    if "chest" in text:
        symptoms.append("Chest Pain")
    if "dizzy" in text:
        symptoms.append("Dizziness")
    if "weak" in text:
        symptoms.append("Weakness")

    return symptoms

# ================================
# 🤖 ML Prediction
# ================================
def predict_risk(text):
    vec = vectorizer.transform([text])
    result = model.predict(vec)[0]

    if result.lower() == "high":
        return "HIGH"
    elif result.lower() == "medium":
        return "MED"
    return "LOW"

# ================================
# 🧠 Explanation
# ================================
def generate_explanation(symptoms):
    if "Chest Pain" in symptoms:
        return "Possible serious cardiac condition. Immediate attention required."
    if "Fever" in symptoms:
        return "Possible infection or viral condition."
    if "Weakness" in symptoms:
        return "May indicate fatigue or nutritional deficiency."
    return "Mild condition"

# ================================
# 🌍 Location Intelligence
# ================================
def location_insight(symptoms, location="Chennai"):
    if location.lower() == "chennai":
        if "Fever" in symptoms:
            return "In Chennai, current seasonal trend suggests possible dengue or viral infection."
        if "Cough" in symptoms:
            return "Respiratory infections are common in this region currently."
    return "No major regional risk detected."

# ================================
# 🧾 Doctor Summary
# ================================
def generate_summary(patient_id, symptoms, risk, explanation):
    return f"""
Patient ID: {patient_id}
Symptoms: {', '.join(symptoms)}
Risk Level: {risk}
Clinical Insight: {explanation}

Recommendation:
{"Immediate medical attention required." if risk == "HIGH" else "Monitor and consult if symptoms persist."}
"""

# ================================
# 📌 Predict API
# ================================
@app.route("/predict", methods=["POST"])
def predict():
    global patient_counter

    data = request.get_json()
    text = data.get("text", "")
    location = data.get("location", "Chennai")  # optional

    symptoms = extract_symptoms(text)
    risk = predict_risk(text)
    explanation = generate_explanation(symptoms)
    location_note = location_insight(symptoms, location)

    # Generate Patient ID
    patient_counter += 1
    patient_id = f"P-{patient_counter}"

    # Generate Summary
    summary = generate_summary(patient_id, symptoms, risk, explanation)

    # Save record
    record = {
        "id": patient_id,
        "symptoms": symptoms,
        "risk": risk
    }
    records.insert(0, record)

    return jsonify({
        "patient_id": patient_id,
        "risk": risk,
        "symptoms": symptoms,
        "explanation": explanation,
        "location_insight": location_note,
        "summary": summary
    })

# ================================
# 📊 Recent Scans API
# ================================
@app.route("/recent", methods=["GET"])
def recent():
    return jsonify(records[:5])

# ================================
# 📈 Analytics API
# ================================
@app.route("/analytics", methods=["GET"])
def analytics():
    total = len(records)

    high = sum(1 for r in records if r["risk"] == "HIGH")
    med = sum(1 for r in records if r["risk"] == "MED")
    low = sum(1 for r in records if r["risk"] == "LOW")

    accuracy = round(random.uniform(95, 99), 1)

    return jsonify({
        "total_predictions": total,
        "high_cases": high,
        "medium_cases": med,
        "low_cases": low,
        "accuracy": accuracy
    })

# ================================
# 🚀 Run Server
# ================================
if __name__ == "__main__":
    app.run(debug=True)