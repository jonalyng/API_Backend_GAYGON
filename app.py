# app.py

from flask import Flask, request, jsonify
import pandas as pd
import joblib

app = Flask(__name__)

# Load model and metadata
model = joblib.load("trained_data/student_pass_fail_model.pkl")
feature_columns = joblib.load("trained_data/model_features.pkl")
label_encoder = joblib.load("trained_data/pass_fail_encoder.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # Expecting JSON input

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # One-hot encode to match training format
    df_encoded = pd.get_dummies(df)
    df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

    # Predict
    prediction_encoded = model.predict(df_encoded)[0]
    prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]

    return jsonify({"prediction": prediction_label})

if __name__ == '__main__':
    app.run(debug=True)
