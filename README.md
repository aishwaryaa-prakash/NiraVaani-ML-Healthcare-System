# NiraVaani – AI-Powered Voice Healthcare Assistant

##  Overview

NiraVaani is an AI-powered healthcare assistant designed to bridge the communication gap between patients and doctors, especially in rural and semi-urban regions.

It converts unstructured natural language input into structured medical insights using Machine Learning and Natural Language Processing (NLP).

---

##  Problem Statement

In many parts of India, particularly rural and semi-urban areas, patients face significant challenges in accessing digital healthcare services due to:

### Low Literacy

* Many patients are unable to read or write effectively
* Difficulty in typing symptoms into applications

### Language Barriers

* Patients communicate in regional languages or informal expressions
* Most healthcare platforms are English-based

### Digital Complexity

* Existing systems require navigation skills and medical understanding
* Not suitable for elderly or first-time users

### Unstructured Symptom Description

Patients describe symptoms informally, for example:

* “Body weak aa irukku”
* “Konjam fever maari irukku”

These inputs are not medically structured and are difficult for systems to interpret.

---

##  Consequences

* Incomplete or unclear information reaches doctors
* Increased chances of misdiagnosis
* Delay in treatment
* Higher health risks

---

##  Proposed Solution

NiraVaani acts as an AI-powered clinical interpretation layer between patient and doctor**.

It enables users to provide symptoms in natural language and converts them into structured, meaningful medical insights.

---

##  System Architecture

1. User Input (Text / Voice Simulation)
2. NLP Processing (Symptom & Context Extraction)
3. Feature Transformation (CountVectorizer)
4. ML Model (Decision Tree Classifier)
5. Risk Prediction (Low / Medium / High)
6. Explainable AI (Reasoning for prediction)
7. Structured Output (Doctor-ready summary)

---

##  Key Features

###  Context Understanding Engine

* Identifies multiple related symptoms from natural sentences
* Example:
  Input: “I feel weak and not eating properly”
  Output: Weakness + Appetite Loss

---

###  NLP-Based Information Extraction

* Extracts:

  * Symptoms
  * Duration

---

###  Machine Learning Risk Prediction

* Uses:

  * CountVectorizer (text → numerical features)
  * Decision Tree Classifier
* Classifies cases into:

  * Low
  * Medium
  * High risk

---

### Explainable AI (XAI)

* Provides reasoning for predictions
* Example:

  * “High risk because chest pain may indicate a serious condition”

---

###  Structured Medical Output

Generates doctor-ready summary:

```json
{
  "symptoms": ["Weakness", "Appetite Loss"],
  "duration": "Not specified",
  "risk": "high",
  "explanation": "Possible nutritional or underlying health issue"
}
```

---

### Flask Backend API

* Endpoint: `/predict`
* Accepts JSON input
* Returns structured output

---

##  Tech Stack

* Frontend: HTML / CSS (Basic UI)
* Backend: Python (Flask)
* Machine Learning:

  * scikit-learn
  * Decision Tree Classifier
  * CountVectorizer
    
* NLP Techniques:

  * Keyword extraction
  * Regex-based pattern matching
  Tools:
  * Postman (API testing)
  * GitHub (version control)


## How to Run the Project

### Step 1: Navigate to ML folder

```bash
cd ml_model
```

### Step 2: Activate virtual environment

```bash
source venv/bin/activate
```

### Step 3: Train the model

```bash
python train_model.py
```

### Step 4: Run backend server

```bash
cd ../backend
python app.py
```

### Step 5: Test API using Postman

* URL: `http://127.0.0.1:5000/predict`
* Method: POST
* Body:

```json
{
  "text": "fever for 3 days"
}
```

---

## Future Enhancements

*  Voice-based input (Speech-to-Text)
*  Multilingual support
*  Personal health memory (history tracking)
*  Hyper-local disease intelligence
*  Doctor assist dashboard
*  Advanced conversational AI

##  Conclusion

NiraVaani demonstrates how AI can simplify healthcare accessibility by transforming unstructured human language into structured clinical insights, improving communication, and enabling faster, more accurate decision-making.

It serves as a strong foundation for building scalable, real-world healthcare solutions.

