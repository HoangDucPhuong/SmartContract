from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
import joblib, json
from utils import normalize_solidity_code

# Load data
with open("train_data.json", "r") as f:
    data = json.load(f)

codes = [normalize_solidity_code(item["Source_Code"]) for item in data]
labels = [[vuln['Type'] for vuln in item["Vulnerabilities"]] for item in data]

# Vectorize code
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(codes)

# Encode labels
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(labels)

# Train model
clf = RandomForestClassifier()
clf.fit(X, y)

# Save artifacts
joblib.dump(clf, "model/rf_model.joblib")
joblib.dump(vectorizer, "model/tfidf_vectorizer.joblib")
joblib.dump(mlb, "model/label_binarizer.joblib")
