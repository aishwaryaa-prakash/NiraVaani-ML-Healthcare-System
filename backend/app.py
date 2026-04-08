from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import re

app = Flask(__name__)
CORS(app)

model = pickle.load(open("../ml_model/model.pkl", "rb"))
vectorizer = pickle.load(open("../ml_model/vectorizer.pkl", "rb"))

history = []

def extract_info(text):
    text = text.lower()
    symptoms = []

    if "fever" in text:
        symptoms.append("Fever")
    if "pain" in text:
        symptoms.append("Pain")
    if "dizzy" in text:
        symptoms.append("Dizziness")
    if "weak" in text:
        symptoms.append("Weakness")

    duration = re.findall(r'\d+', text)
    duration = duration[0] + " days" if duration else "Not specified"

    return symptoms, duration

def predict_risk(text):
    vec = vectorizer.transform([text])
    return model.predict(vec)[0]

def explanation(symptoms):
    if "Pain" in symptoms:
        return "Possible serious condition"
    return "Mild condition"

def history_check(text):
    history.append(text)
    return "Recurring" if history.count(text) > 1 else "First time"

def followup(duration):
    if duration == "Not specified":
        return "Please specify duration"
    return "Monitor condition"

@app.route("/predict", methods=["POST"])
def predict():
    text = request.get_json()["text"]

    symptoms, duration = extract_info(text)
    risk = predict_risk(text)

    return jsonify({
        "symptoms": symptoms,
        "duration": duration,
        "risk": risk,
        "explanation": explanation(symptoms),
        "history_flag": history_check(text),
        "follow_up": followup(duration)
    })

if __name__ == "__main__":
    app.run(debug=True)