from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load the Scikit-Learn model
with open('Has_Laptop_or_Not.pkl', 'rb') as file:
    model = pickle.load(file)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Extract the 5 required features
        # Note: Ensure the incoming data handles categorical encoding if needed
        age = float(data.get('Age', 0))
        gender = float(data.get('Gender', 0))
        region = float(data.get('Region', 0))
        occupation = float(data.get('Occupation', 0))
        income = float(data.get('Income', 0))

        # Format features for the Decision Tree
        features = np.array([[age, gender, region, occupation, income]])

        # Make prediction
        prediction = model.predict(features)
        
        return jsonify({
            'status': 'success',
            'prediction': prediction[0] # Will return 'yes' or 'no'
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 400

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy'})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
