from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import os

app = Flask(__name__)

db_path = os.path.join(os.path.dirname(__file__), 'wine_predictions.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


class Prediction(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    f1 = db.Column(db.Float, nullable=False)
    f2 = db.Column(db.Float, nullable=False)
    f3 = db.Column(db.Float, nullable=False)
    f4 = db.Column(db.Float, nullable=False)
    f5 = db.Column(db.Float, nullable=False)
    predicted_class = db.Column(db.String(50), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'f1': self.f1,
            'f2': self.f2,
            'f3': self.f3,
            'f4': self.f4,
            'f5': self.f5,
            'predicted_class': self.predicted_class,
            'timestamp': self.timestamp.isoformat()
        }


with app.app_context():
    db.create_all()


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'service': 'dbapp'}), 200


@app.route('/predictions', methods=['POST'])
def save_prediction():
    try:
        data = request.get_json()
        prediction = Prediction(
            f1=data['f1'],
            f2=data['f2'],
            f3=data['f3'],
            f4=data['f4'],
            f5=data['f5'],
            predicted_class=data['predicted_class']
        )
        db.session.add(prediction)
        db.session.commit()
        return jsonify({'message': 'Prediction saved successfully', 'id': prediction.id}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 400


@app.route('/predictions', methods=['GET'])
def get_predictions():
    try:
        predictions = Prediction.query.order_by(Prediction.timestamp.desc()).all()
        return jsonify([pred.to_dict() for pred in predictions]), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predictions/<int:prediction_id>', methods=['GET'])
def get_prediction(prediction_id):
    try:
        prediction = Prediction.query.get_or_404(prediction_id)
        return jsonify(prediction.to_dict()), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 404


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8001, debug=True)


