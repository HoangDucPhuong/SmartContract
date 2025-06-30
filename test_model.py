import json
import joblib
from utils import normalize_solidity_code

# Load model and encoders
model = joblib.load("model/rf_model.joblib")
mlb = joblib.load("model/label_binarizer.joblib")
vectorizer = joblib.load("model/tfidf_vectorizer.joblib")

# Load test data
with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

total = len(data)
correct = 0
mismatches = []

for idx, item in enumerate(data):
    code = normalize_solidity_code(item["Source_Code"])
    actual_vulns = item.get("Vulnerabilities", [])
    actual_labels = [v if isinstance(v, str) else v.get("Type") for v in actual_vulns]

    X = vectorizer.transform([code])
    pred = model.predict(X)

    if pred.shape[0] == 0:
        pred_labels = []
    else:
        pred_labels = mlb.inverse_transform(pred)[0] if pred.shape[1] == len(mlb.classes_) else []

    if set(pred_labels) == set(actual_labels):
        correct += 1
    else:
        mismatches.append({
            "Contract_Index": idx,
            "Predict": list(pred_labels),
            "Actual": actual_labels
        })

accuracy = correct / total if total > 0 else 0

print(f"\n✔ Total Accuracy: {accuracy * 100:.2f}% ({correct}/{total})")
print(f"✘ Mismatched Contracts: {len(mismatches)}")

if mismatches:
    print("\nSample mismatches:")
    for m in mismatches[:5]:
        print(f"Index {m['Contract_Index']}:\n → Predict: {m['Predict']}\n → Actual: {m['Actual']}\n")
