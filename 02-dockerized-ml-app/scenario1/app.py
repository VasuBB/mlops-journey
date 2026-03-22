from flask import Flask, render_template, jsonify, request
import os
import pickle
import numpy as np

app = Flask(__name__)


@app.route("/")
def index():
    return render_template("index.html")


def load_model():
    model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
    if not os.path.exists(model_path):
        return None
    with open(model_path, "rb") as f:
        return pickle.load(f)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        f1 = float(request.form["feature1"])  # alcohol
        f2 = float(request.form["feature2"])  # malic_acid
        f3 = float(request.form["feature3"])  # ash
        f4 = float(request.form["feature4"])  # alcalinity_of_ash
        f5 = float(request.form["feature5"])  # magnesium

        model = load_model()
        if model is None:
            return render_template("index.html", error="model.pkl not found. Run train_model.py."), 200

        X = np.array([[f1, f2, f3, f4, f5]])
        pred_idx = int(model.predict(X)[0])
        return render_template("index.html", prediction=str(pred_idx), f1=f1, f2=f2, f3=f3, f4=f4, f5=f5), 200
    except ValueError:
        return render_template("index.html", error="Invalid input. Enter numeric values."), 200
    except Exception as e:
        return render_template("index.html", error=str(e)), 200


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "scenario1"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)


