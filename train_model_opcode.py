import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer

# Load and preprocess training data
with open("train_data.json", "r") as file:
    data = json.load(file)

X = [" ".join(item["Opcode"]) if isinstance(item["Opcode"], list) else str(item["Opcode"]) for item in data]
y = [list(set(v["Type"] for v in item["Vulnerabilities"])) for item in data]

# Vectorize the opcode input
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

# Encode the labels
mlb = MultiLabelBinarizer()
y_bin = mlb.fit_transform(y)

# Train the model
model = RandomForestClassifier()
model.fit(X_vec, y_bin)

# Save components
joblib.dump(model, "model/opcode_model.joblib")
joblib.dump(mlb, "model/opcode_label_binarizer.joblib")
joblib.dump(vectorizer, "model/opcode_tfidf_vectorizer.joblib")
print("Model and label binarizer saved successfully.")