import re
import joblib
import numpy as np

def normalize_solidity_code(code):
    code = re.sub(r'//.*', '', code)
    code = re.sub(r'/\*[\s\S]*?\*/', '', code)
    code = re.sub(r'\s+', ' ', code)

    keywords = [
        'address', 'uint256', 'require', 'msg', 'sender', 'call', 'value',
        'function', 'public', 'private', 'external', 'internal', 'view', 'returns'
    ]
    keywords_pattern = r'\b(?:' + '|'.join(re.escape(k) for k in keywords) + r')\b'

    tokens = re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', code)
    unique_tokens = set(tokens)
    replace_map = {}
    counter = 1

    for token in unique_tokens:
        if not re.fullmatch(keywords_pattern, token):
            replace_map[token] = f'VAR{counter}'
            counter += 1

    for original, replacement in replace_map.items():
        code = re.sub(rf'\b{original}\b', replacement, code)

    return code.strip()

# Load model, vectorizer, and label binarizer
rf_model = joblib.load("model/rf_model.joblib")
tfidf_vectorizer = joblib.load("model/tfidf_vectorizer.joblib")
label_binarizer = joblib.load("model/label_binarizer.joblib")

def predict_vulnerabilities(contract_code):
    normalized_code = normalize_solidity_code(contract_code)
    tfidf_features = tfidf_vectorizer.transform([normalized_code])
    prediction = rf_model.predict(tfidf_features)
    confidence_scores = rf_model.predict_proba(tfidf_features)

    predicted_labels = label_binarizer.inverse_transform(prediction)
    result = {
        "predicted_labels": predicted_labels[0] if predicted_labels else [],
        "confidence": float(confidence_scores[0].max()) if confidence_scores else 0.0
    }
    return result
