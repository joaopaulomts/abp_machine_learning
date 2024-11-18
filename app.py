from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Carregar o modelo treinado
model = joblib.load('random_forest_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()  # Receber os dados em formato JSON
    features = np.array([data['CO AQI Value'], data['Ozone AQI Value'], data['NO2 AQI Value'], data['PM2.5 AQI Value']]).reshape(1, -1)
    prediction = model.predict(features)
    return jsonify({'prediction': prediction[0]})

if __name__ == '__main__':
    app.run(debug=True)
