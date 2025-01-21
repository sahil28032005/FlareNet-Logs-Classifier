from flask import Flask, request, jsonify
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = joblib.load('deployment_model.pkl')

# Load the scaler (optional, if you're using StandardScaler)
scaler = StandardScaler()


@app.route('/health')
def health():
    return "Deployment Prediction Model is running!"


if __name__ == '__main__':
    app.run(debug=True)
