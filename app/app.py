from flask import Flask, request, jsonify
import joblib
import pandas as pd
# from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the saved model
model = joblib.load('deployment_model.pkl')

# Load the scaler (optional, if you're using StandardScaler)
# scaler = StandardScaler()

#health route for test
@app.route('/health')
def health():
    return "Deployment Prediction Model is running!"

#predictor route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the POST request
        data = request.get_json()

        # Convert JSON input to DataFrame
        input_data = pd.DataFrame(data)

        # Process the `timestamp` field (if present)
        if 'timestamp' in input_data.columns:
            input_data['timestamp'] = pd.to_datetime(input_data['timestamp'], errors='coerce')
            input_data['hour'] = input_data['timestamp'].dt.hour
            input_data['day_of_week'] = input_data['timestamp'].dt.dayofweek
            input_data['month'] = input_data['timestamp'].dt.month
            input_data.drop(columns=['timestamp'], inplace=True)

        # Select relevant columns and preprocess
        features = input_data.drop(columns=['deployment_id', 'event_id', 'file_name', 'log_level'], errors='ignore')

        # Scale numeric features
        # features[['file_size_in_bytes', 'time_taken']] = scaler.transform(features[['file_size_in_bytes', 'time_taken']])

        # Make predictions
        predictions = model.predict(features)

        # Prepare the response
        response = {
            'predictions': predictions.tolist(),
            'deployment_id': input_data['deployment_id'].tolist() if 'deployment_id' in input_data.columns else None
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
