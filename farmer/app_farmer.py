# app.py
from flask import Flask, request, jsonify, render_template
import joblib
import pandas as pd
import re

app = Flask(__name__)
model = joblib.load('model_farmer.joblib')

FEATURES = ["N","P","K","temperature","humidity","ph","rainfall"]
PATTERN = re.compile(r"\b(N|P|K|temperature|humidity|ph|rainfall)\s*[:=]?\s*([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)

def parse_input(text):
    return {m.group(1).lower(): float(m.group(2)) for m in PATTERN.finditer(text)}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/parse', methods=['POST'])
def parse():
    feats = parse_input(request.json.get('text',''))
    return jsonify(feats)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    df = pd.DataFrame([data])
    pred = model.predict(df.values)[0]
    return jsonify({'prediction': int(pred)})

if __name__ == '__main__':
    app.run(debug=True)
