from flask import Flask, request, render_template, jsonify
import pickle
import numpy as np
from waitress import serve

app = Flask(__name__, template_folder='templates')

# Load model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = model_data['class_names']


@app.route('/')
def home():
    return render_template('index_api.html', title="Wine Predictor API")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form
        alcohol = float(data['alcohol'])
        flavanoids = float(data['flavanoids'])
        color_intensity = float(data['color_intensity'])
        hue = float(data['hue'])
        proline = float(data['proline'])

        features = np.array([[alcohol, flavanoids, color_intensity, hue, proline]])
        scaled = scaler.transform(features)
        prediction = model.predict(scaled)
        result = class_names[prediction[0]]

        return jsonify({"predicted_class": result})
    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == "__main__":
    serve(app, host="0.0.0.0", port=5002)
