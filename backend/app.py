from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

# Load model
model = pickle.load(open("../ml_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../ml_model/vectorizer.pkl", "rb"))

# -------------------------------
# 🧠 In-memory history (session-level)
# -------------------------------
history = []

# -------------------------------
# 🔍 NLP - Extract Info
# -------------------------------
def extract_info(text):
    text = text.lower()
    symptoms = []

    # Basic symptoms
    if "fever" in text:
        symptoms.append("Fever")
    if "cough" in text:
        symptoms.append("Cough")
    if "pain" in text:
        symptoms.append("Pain")
    if "chest" in text:
        symptoms.append("Chest Pain")

    # Extended context understanding
    if "weak" in text or "tired" in text:
        symptoms.append("Weakness")
    if "not eating" in text or "no appetite" in text:
        symptoms.append("Appetite Loss")
    if "vomit" in text:
        symptoms.append("Vomiting")
    if "dizzy" in text:
        symptoms.append("Dizziness")
    if "headache" in text:
        symptoms.append("Headache")

    # Duration extraction
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
        return "Moderate risk due to prolonged fever."
    
    if "Weakness" in symptoms and "Appetite Loss" in symptoms:
        return "Possible nutritional deficiency or underlying health condition."
    
    if "Vomiting" in symptoms:
        return "Risk due to possible dehydration or infection."
    
    return "Low risk based on mild or non-critical symptoms."

# -------------------------------
# 📊 History Tracking
# -------------------------------
def check_history(text):
    history.append(text)
    
    if history.count(text) > 1:
        return "Yes - recurring symptom detected"
    
    return "No"

# -------------------------------
# 💬 Follow-up Suggestions
# -------------------------------
def generate_followup(duration, symptoms):
    if duration == "Not specified":
        return "Please mention how long you have been experiencing these symptoms."
    
    if "Chest Pain" in symptoms:
        return "Immediate medical attention is recommended."
    
    return "Monitor symptoms and consult a doctor if condition worsens."

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
    history_flag = check_history(text)
    follow_up = generate_followup(duration, symptoms)

    return jsonify({
        "symptoms": symptoms,
        "duration": duration,
        "risk": risk,
        "explanation": explanation,
        "history_flag": history_flag,
        "follow_up": follow_up
    })

# -------------------------------
# 🚀 Run Server
# -------------------------------
if __name__ == "__main__":
    app.run(debug=True)