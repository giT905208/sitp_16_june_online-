from flask import Flask, render_template, request
import numpy as np
from joblib import load

app = Flask(__name__)

# Load your trained model
model = load("model_insurance.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        try:
            age = int(request.form["age"])
            sex = int(request.form["sex"])  # 0 or 1
            bmi = float(request.form["bmi"])
            children = int(request.form["children"])
            smoker = int(request.form["smoker"])  # 0 or 1
            region = int(request.form["region"])  # encoded value
        except ValueError:
            return "Please enter valid input values."

        input_data = np.array([[age, sex, bmi, children, smoker, region]])
        prediction = model.predict(input_data)[0]
        return render_template("index.html", prediction_text=f"Predicted Insurance Charge: ₹{prediction:.2f}")

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
