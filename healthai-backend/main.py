import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib
import os
from lime.lime_tabular import LimeTabularExplainer
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Enable CORS for frontend communication

# Define the directory where your models and data are stored
MODEL_DIR = 'models/'

# Load the models and data
try:
    rf = joblib.load(os.path.join(MODEL_DIR, 'random_forest_model.joblib'))
    knn = joblib.load(os.path.join(MODEL_DIR, 'knn_model.joblib'))
    FEATURES = joblib.load(os.path.join(MODEL_DIR, 'features.joblib'))
    label_names = joblib.load(os.path.join(MODEL_DIR, 'label_names.joblib'))
    X_train = joblib.load(os.path.join(MODEL_DIR, 'X_train.joblib'))
    print("✅ Models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"❌ Error loading files: {e}. Please ensure all .joblib files are in the 'models' directory.")
    exit()

# RECREATE the LIME explainer here, since it can't be pickled
lime_explainer = LimeTabularExplainer(
    training_data=X_train,
    feature_names=FEATURES,
    class_names=label_names,
    mode="classification"
)
print("✅ LIME Explainer recreated.")

def calculate_risk_scores(cbc):
    hgb = float(cbc.get("Hemoglobin", 0))
    wbc = float(cbc.get("WBC", 0))
    nlr = float(cbc.get("NLR", 0))

    anemia_risk = max(0, min(100, (15 - hgb) * 10))
    infection_risk = max(0, min(100, (wbc - 11) * 10))
    cardio_risk = max(0, min(100, nlr * 10))
    leukemia_risk = max(0, min(100, (wbc - 18) * 20))

    return {
        "Anemia Risk": round(anemia_risk, 2),
        "Infection Risk": round(infection_risk, 2),
        "Cardiovascular Risk": round(cardio_risk, 2),
        "Leukemia Risk": round(leukemia_risk, 2),
    }

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    cbc_data = pd.DataFrame([data], columns=FEATURES)

    alpha = 0.80
    prob_rf = rf.predict_proba(cbc_data)
    prob_knn = knn.predict_proba(cbc_data.values)
    fused_prob = alpha * prob_rf + (1 - alpha) * prob_knn

    predicted_class_code = int(np.argmax(fused_prob))
    predicted_label = label_names[predicted_class_code]

    risk_scores = calculate_risk_scores(data)

    # Generate LIME explanation
    exp = lime_explainer.explain_instance(
        cbc_data.values[0],
        knn.predict_proba,
        num_features=len(FEATURES)
    )
    explanation = exp.as_list()

    response = {
        'prediction': predicted_label,
        'probabilities': {label_names[i]: float(p) for i, p in enumerate(fused_prob[0])},
        'risk_scores': risk_scores,
        'explanation': explanation
    }

    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)