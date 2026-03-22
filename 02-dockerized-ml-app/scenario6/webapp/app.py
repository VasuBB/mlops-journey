from flask import Flask, render_template, request
import pickle
import numpy as np
import requests
import os
from sklearn.datasets import load_wine

app = Flask(__name__)

model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
if os.path.exists(model_path):
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
else:
    model = None
    print("Warning: model.pkl not found. Please run train_model.py first.")

wine = load_wine()
class_names = wine.target_names

DBAPP_URL = os.getenv('DBAPP_URL', 'http://dbapp:8001')


def predict_wine(f1, f2, f3, f4, f5):
    if model is None:
        return None, "Model not loaded"
    features = np.array([[f1, f2, f3, f4, f5]])
    prediction = int(model.predict(features)[0])
    class_name = class_names[prediction]
    return class_name, None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        f1 = float(request.form['feature1'])
        f2 = float(request.form['feature2'])
        f3 = float(request.form['feature3'])
        f4 = float(request.form['feature4'])
        f5 = float(request.form['feature5'])

        class_name, error = predict_wine(f1, f2, f3, f4, f5)
        if error:
            return render_template('index.html', error=error)

        try:
            db_response = requests.post(
                f'{DBAPP_URL}/predictions',
                json={'f1': f1, 'f2': f2, 'f3': f3, 'f4': f4, 'f5': f5, 'predicted_class': class_name},
                timeout=5
            )
            if db_response.status_code != 201:
                print(f"Warning: Failed to save prediction to database: {db_response.text}")
        except Exception as e:
            print(f"Warning: Database connection failed: {str(e)}")

        return render_template(
            'index.html',
            prediction=class_name,
            f1=f1,
            f2=f2,
            f3=f3,
            f4=f4,
            f5=f5
        )
    except ValueError:
        return render_template('index.html', error='Invalid input. Please enter numeric values.')
    except Exception as e:
        return render_template('index.html', error=f'An error occurred: {str(e)}')


@app.route('/records')
def display_records():
    try:
        response = requests.get(f'{DBAPP_URL}/predictions', timeout=5)
        if response.status_code == 200:
            predictions = response.json()
            return render_template('display_records.html', predictions=predictions)
        else:
            return render_template('display_records.html', error=f'Failed to fetch records: {response.text}', predictions=[])
    except Exception as e:
        return render_template('display_records.html', error=f'Database connection failed: {str(e)}', predictions=[])


@app.route('/health', methods=['GET'])
def health_check():
    return {'status': 'healthy', 'service': 'webapp', 'model_loaded': model is not None}, 200


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)


