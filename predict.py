import joblib
import numpy as np
from flask import Flask, request, jsonify, render_template
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

model = joblib.load('heart_disease_svm_model.pkl')

scaler = joblib.load('scaler.pkl')


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = [float(x) for x in request.form.values()]
    
    features = np.array(user_input).reshape(1, -1)

    scaled_features = scaler.transform(features)
    
    prediction = model.predict(scaled_features)[0]
    
    if prediction == 0:
        result = 'No Heart Disease Detected'
    else:
        result = 'Heart Disease Detected'

    return render_template('result.html', prediction_text=f'Result: {result}')

if __name__ == "__main__":
    app.run(debug=True)
