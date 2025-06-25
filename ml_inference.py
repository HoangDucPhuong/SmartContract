# ml_inference.py

import joblib
import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load pre-trained model components
model_dir = os.path.join(os.path.dirname(__file__), 'model')

model_path = os.path.join(model_dir, 'rf_model.joblib')
label_binarizer_path = os.path.join(model_dir, 'label_binarizer.joblib')
tfidf_vectorizer_path = os.path.join(model_dir, 'tfidf_vectorizer.joblib')

rf_model = joblib.load(model_path)
label_binarizer = joblib.load(label_binarizer_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)

def predict_vulnerabilities(code_text):
    """
    Takes Solidity source code as input and returns predicted vulnerability labels.
    """
    # Transform input
    vectorized_input = tfidf_vectorizer.transform([code_text])

    # Predict labels
    predictions = rf_model.predict(vectorized_input)
    predicted_labels = label_binarizer.inverse_transform(predictions)

    # Handle multilabel probabilities
    confidence_scores = []
    if hasattr(rf_model, "predict_proba"):
        probas = rf_model.predict_proba(vectorized_input)
        for i in range(len(predictions[0])):
            # Only collect scores for predicted = 1 labels
            if predictions[0][i] == 1:
                proba_array = probas[i]  # shape: (1,2) or (1,) depending
                if hasattr(proba_array, '__getitem__') and len(proba_array[0]) > 1:
                    confidence_scores.append(float(proba_array[0][1]))
                else:
                    confidence_scores.append(float(proba_array[0]))  # fallback
    return {
        "predicted_labels": predicted_labels[0] if predicted_labels else [],
        "confidence_scores": confidence_scores
    }