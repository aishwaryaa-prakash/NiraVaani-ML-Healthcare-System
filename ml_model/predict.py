import pickle
import re

# Load model
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Simple symptom extraction
def extract_info(text):
    symptoms = []
    
    if "fever" in text:
        symptoms.append("Fever")
    if "cough" in text:
        symptoms.append("Cough")
    if "pain" in text:
        symptoms.append("Pain")
    if "chest" in text:
        symptoms.append("Chest Pain")

    # Extract duration (numbers)
    duration = re.findall(r'\d+', text)
    duration = duration[0] + " days" if duration else "Not specified"

    return symptoms, duration


def predict_risk(user_input):
    input_vector = vectorizer.transform([user_input])
    prediction = model.predict(input_vector)
    return prediction[0]


# Run
text = input("Enter symptoms: ")

symptoms, duration = extract_info(text)
risk = predict_risk(text)

print("\n--- Patient Summary ---")
print("Symptoms:", ", ".join(symptoms) if symptoms else "Unknown")
print("Duration:", duration)
print("Risk Level:", risk)