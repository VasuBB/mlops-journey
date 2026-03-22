
from flask import Flask, request, render_template
import pickle
import numpy as np
from waitress import serve

app = Flask(__name__, template_folder='templates')

# Load the trained model
with open('model.pkl', 'rb') as f:
    model_data = pickle.load(f)
    model = model_data['model']
    scaler = model_data['scaler']
    class_names = model_data['class_names']

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data - 5 features
        alcohol = float(request.form['alcohol'])
        flavanoids = float(request.form['flavanoids'])
        color_intensity = float(request.form['color_intensity'])
        hue = float(request.form['hue'])
        proline = float(request.form['proline'])

        # Prepare input
        features = np.array([[alcohol, flavanoids, color_intensity, hue, proline]])
        
        # Scale features
        features_scaled = scaler.transform(features)

        # Predict
        prediction = model.predict(features_scaled)
        predicted_class = class_names[prediction[0]]
        print(predicted_class)
        return render_template(
            'index.html',
            prediction_text=f"Predicted Wine Class: {predicted_class}"
        )
    except Exception as e:
        return render_template('index.html', prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":


    serve(app, host="0.0.0.0", port=5001)
