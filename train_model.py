import json
import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Load dataset
with open("draft.json", "r", encoding="utf-8") as f:
    data = json.load(f)

texts = []
labels = []

# Extract source code and vulnerability types
for item in data:
    code = item.get("Source_Code", "").strip()
    vulnerabilities = item.get("Vulnerabilities", [])

    if not code or not vulnerabilities:
        continue

    vuln_types = list({v.get("Type", "").strip() for v in vulnerabilities if v.get("Type")})

    if not vuln_types:
        continue

    texts.append(code)
    labels.append(vuln_types)

# Vectorize source code
vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_features=10000, stop_words='english')
X = vectorizer.fit_transform(texts)

# Binarize multi-label targets
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=mlb.classes_))

# Save components
model_dir = os.path.join(os.path.dirname(__file__), 'model')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, 'rf_model.joblib'))
joblib.dump(mlb, os.path.join(model_dir, 'label_binarizer.joblib'))
joblib.dump(vectorizer, os.path.join(model_dir, 'tfidf_vectorizer.joblib'))

print("Model training complete and saved to 'model' directory.")
