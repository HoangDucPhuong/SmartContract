import json
import joblib
import os
from sklearn.metrics import classification_report
from ml_inference import predict_vulnerabilities

# Load test data
with open("test_data.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)

correct = 0
total = len(test_data)

print("=== TEST RESULTS ===\n")

for item in test_data:
    code = item["Source_Code"]
    actual = set(v["Type"] for v in item["Vulnerabilities"] if v.get("Type"))

    result = predict_vulnerabilities(code)
    predicted = set(result["predicted_labels"])
    confidence = result["confidence_scores"]

    print("Contract:", code[:50].replace("\n", " ") + ("..." if len(code) > 50 else ""))
    print("Predicted:", predicted)
    print("Actual:", actual)
    print("Confidence Scores:", confidence)

    if predicted == actual:
        correct += 1
    print("-" * 50)

print("\n=== SUMMARY ===")
print(f"Total Contracts Tested : {total}")
print(f"Correct Predictions    : {correct}")
print(f"Accuracy               : {correct / total * 100:.2f}%")
print(f"Wrong Predictions      : {total - correct}")
