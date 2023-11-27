from flask import Flask, request, jsonify
from joblib import load
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the models and label encoders
model_harvest_info = load('harvest_prediction_model.joblib')
scaler = load('standard_scaler.joblib')

model_harvest = model_harvest_info['model']
label_encoder_label = model_harvest_info['label_encoder_label']
label_encoder_country = model_harvest_info['label_encoder_country']
label_encoder_harvest_season = model_harvest_info['label_encoder_harvest_season']

# Define API endpoint
@app.route('/predict', methods=['GET'])
def predict():
    try:
        # Sample input values
        sample_input = {
            'temperature': float(request.args.get('temperature')),
            'humidity': float(request.args.get('humidity')),
            'ph': float(request.args.get('ph')),
            'water_availability': float(request.args.get('water_availability')),
            'label': request.args.get('label'),
            'country': request.args.get('country')
        }

        # Use the loaded label encoders
        label_encoded = label_encoder_label.transform([sample_input['label']])[0]
        country_encoded = label_encoder_country.transform([sample_input['country']])[0]

        # Standardize the sample input
        sample_input_scaled = scaler.transform([[
            sample_input['temperature'],
            sample_input['humidity'],
            sample_input['ph'],
            sample_input['water_availability'],
            label_encoded,
            country_encoded,
        ]])

        # Make predictions for harvest season
        harvest_season_prediction = model_harvest.predict(sample_input_scaled)[0]
        predicted_harvest_season = label_encoder_harvest_season.inverse_transform([harvest_season_prediction])[0]

        result = {
            "Best_Harvest_Season": predicted_harvest_season
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)})

# Run the Flask application
if __name__ == '__main__':
    app.run(debug=True)
    # app.run(host='0.0.0.0', port=8080)
