from ml_inference import predict_vulnerabilities
import json

with open("draft.json", "r", encoding="utf-8") as f:
    data = json.load(f)

correct = 0
total = 0

for item in data[:10]:  # limit to first 10 entries
    code = item.get("Source_Code", "").strip()
    vulnerabilities = item.get("Vulnerabilities", [])

    if not code or not vulnerabilities:
        continue

    # Extract only the 'Type' of each vulnerability
    actual_labels = set(
        v["Type"].strip() for v in vulnerabilities if v.get("Type")
    )

    # Run inference
    result = predict_vulnerabilities(code)
    predicted_labels = set(result["predicted_labels"])

    print("Smart Contract:", code[:100], "...")
    print("Predicted:", predicted_labels)
    print("Actual:   ", actual_labels)
    print("Confidence Scores:", result["confidence_scores"])
    print("-" * 60)

    if predicted_labels & actual_labels:
        correct += 1
    total += 1

# Print test accuracy
accuracy = correct / total if total > 0 else 0
print(f"Overlap Accuracy: {accuracy:.2f}")
