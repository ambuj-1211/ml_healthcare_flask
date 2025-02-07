import pickle

import numpy as np
import preprocessor as preproc
from flask import Flask, jsonify, request
from flask_cors import CORS

pp = preproc.preprocessor()
model = pickle.load(open('latest_model.pkl', 'rb'))

app = Flask(__name__)
CORS(app)

@app.route('/', methods= ['GET'])
def first_page():
    return f"Your api is working properly."

@app.route('/predict', methods= ['POST','GET'])
def predict_disease():
    
    if request.method == 'POST':
        try:
            data = request.get_json()
            symptoms = data.get('symptom', '')
            symptoms = pp.forward(symptoms)
            my_prediction = model.predict(symptoms)
            my_prediction_str = ' '.join(my_prediction.astype(str))
            return jsonify({'prediction': my_prediction_str}), 200
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    return jsonify({'error': 'Invalid request method'}), 405

if __name__ == '__main__':
    app.run(debug=True,host="0.0.0.0")