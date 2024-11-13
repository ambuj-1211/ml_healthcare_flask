import pickle

import numpy as np
import predictor as predict  # Ensure the correct import path
import preprocessor as preproc  # Ensure the correct import path
from flask import Flask, jsonify, request
from flask_cors import CORS

pp = preproc.preprocessor()
model = pickle.load(open('model.pkl', 'rb'))

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/predict', methods= ['POST','GET'])
def predict_disease():
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            symptoms = data.get('symptom', '')
            symptoms = pp.forward(symptoms)
            my_prediction = model.predict(symptoms)
            my_prediction_str = ' '.join(my_prediction.astype(str))
            return jsonify({'prediction': my_prediction_str}), 200  # Return a 200 status code for success
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True)