from flask import Flask, request, jsonify
import pickle
import re

app = Flask(__name__)

# Load ML model
model = pickle.load(open("../ml_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../ml_model/vectorizer.pkl", "rb"))

# -------------------------------
# 🔍 Symptom Extraction (NLP)
# -------------------------------
def extract_info(text):
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
    if "weak" in text:
        symptoms.append("Weakness")
    if "not eating" in text or "no appetite" in text:
        symptoms.append("Appetite Loss")

    # Extract duration
    duration_match = re.findall(r'\d+', text)
    duration = duration_match[0] + " days" if duration_match else "Not specified"

    return symptoms, duration


# -------------------------------
# 🤖 ML Prediction
# -------------------------------
def predict_risk(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    return prediction[0]


# -------------------------------
# 🧠 Explainable AI
# -------------------------------
def generate_explanation(symptoms, duration, risk):
    if "Chest Pain" in symptoms:
        return "High risk because chest pain may indicate a serious cardiac condition."
    
    if "Fever" in symptoms and "3" in duration:
        return "Moderate risk due to prolonged fever over multiple days."
    
    if "Weakness" in symptoms and "Appetite Loss" in symptoms:
        return "Possible nutritional or underlying health issue due to weakness and appetite loss."
    
    return "Low risk based on mild or less critical symptoms."


# -------------------------------
# 🌐 API Endpoint
# -------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    text = data.get("text", "")

    symptoms, duration = extract_info(text)
    risk = predict_risk(text)
    explanation = generate_explanation(symptoms, duration, risk)

    return jsonify({
        "symptoms": symptoms,
        "duration": duration,
        "risk": risk,
        "explanation": explanation
    })


# -------------------------------
# 🚀 Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)
    