from flask import Flask, request, jsonify
import os
from ml_inference import predict_vulnerabilities

app = Flask(__name__)

@app.route("/")
def index():
    return jsonify({"message": "Welcome to the Smart Contract Vulnerability Detector API."})

@app.route("/predict", methods=["POST"])
def upload_and_predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "File name is empty"}), 400

    try:
        code = file.read().decode("utf-8")
    except UnicodeDecodeError:
        return jsonify({"error": "Unable to decode file. Make sure it is UTF-8 encoded Solidity code."}), 400

    if not code.strip():
        return jsonify({"error": "File is empty or contains no valid content."}), 400

    result = predict_vulnerabilities(code)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
