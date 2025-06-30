import json
import joblib
from sklearn.metrics import accuracy_score
from utils import decode_labels

# Load test data
with open("test_data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

X_test = [" ".join(item["Opcode"]) if isinstance(item["Opcode"], list) else str(item["Opcode"]) for item in data]
y_test = [list(set(v["Type"] for v in item["Vulnerabilities"])) for item in data]

# Load pre-trained model and tools
model = joblib.load("model/opcode_model.joblib")
vectorizer = joblib.load("model/opcode_tfidf_vectorizer.joblib")
mlb = joblib.load("model/opcode_label_binarizer.joblib")

# Vectorize input
X_test_vec = vectorizer.transform(X_test)

# Predict directly
y_pred_bin = model.predict(X_test_vec)
y_pred = mlb.inverse_transform(y_pred_bin)

# Decode true labels
y_true = decode_labels(y_test, mlb)

# Evaluate
correct = 0
mismatches = []
for idx, (pred, true) in enumerate(zip(y_pred, y_true)):
    if set(pred) == set(true):
        correct += 1
    else:
        mismatches.append((idx, pred, true))

accuracy = correct / len(y_true)
print(f"\n✓ Opcode Accuracy: {accuracy * 100:.2f}% ({correct}/{len(y_true)})")
print(f"✘ Opcode Mismatches: {len(mismatches)}")

if mismatches:
    print("\nSample mismatches:")
    for idx, pred, true in mismatches:
        print(f"Index {idx}:\n➜ Predict: {pred}\n➜ Actual : {true}\n")
