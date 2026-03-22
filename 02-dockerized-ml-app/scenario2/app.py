from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os
import pickle
import numpy as np

app = Flask(__name__)

db_path = os.path.join(os.path.dirname(__file__), "predictions.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{db_path}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    feature1 = db.Column(db.Float, nullable=False)
    feature2 = db.Column(db.Float, nullable=False)
    feature3 = db.Column(db.Float, nullable=False)
    feature4 = db.Column(db.Float, nullable=False)
    feature5 = db.Column(db.Float, nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)


with app.app_context():
    db.create_all()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        feature1 = float(request.form["feature1"])
        feature2 = float(request.form["feature2"])
        feature3 = float(request.form["feature3"])
        feature4 = float(request.form["feature4"])
        feature5 = float(request.form["feature5"])

        model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
        if not os.path.exists(model_path):
            return render_template("index.html", error="model.pkl not found. Run train_model.py.")
        with open(model_path, "rb") as f:
            model = pickle.load(f)

        X = np.array([[feature1, feature2, feature3, feature4, feature5]])
        predicted_class = str(int(model.predict(X)[0]))

        rec = Prediction(
            feature1=feature1,
            feature2=feature2,
            feature3=feature3,
            feature4=feature4,
            feature5=feature5,
            predicted_class=predicted_class,
        )
        db.session.add(rec)
        db.session.commit()

        return render_template(
            "results.html",
            prediction=predicted_class,
            feature1=feature1,
            feature2=feature2,
            feature3=feature3,
            feature4=feature4,
            feature5=feature5,
        )
    except Exception as e:
        return render_template("index.html", error=str(e))


@app.route("/records")
def records():
    items = Prediction.query.order_by(Prediction.timestamp.desc()).all()
    return render_template("results.html", predictions=items)


@app.route("/health")
def health():
    return jsonify({"status": "healthy", "service": "scenario2"})


if __name__ == "__main__":
    port = int(os.getenv("PORT", "5000"))
    app.run(host="0.0.0.0", port=port)


