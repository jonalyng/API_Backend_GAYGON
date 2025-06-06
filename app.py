from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib

app = Flask(__name__)
CORS(app)

# Load model and preprocessing artifacts
try:
    model = joblib.load("trained_data/student_pass_fail_model.pkl")
    feature_columns = joblib.load("trained_data/model_features.pkl")
    label_encoder = joblib.load("trained_data/pass_fail_encoder.pkl")
    print("Model and encoders loaded successfully.")
except Exception as e:
    print("Model loading error:", e)
    model = None

@app.route('/predict', methods=['POST'])
def predict():
    if not model:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json

        # Expected keys from the frontend
        expected_keys = [
            "Gender", "Study_Hours_per_Week", "Attendance_Rate",
            "Past_Exam_Scores", "Parental_Education_Level",
            "Internet_Access_at_Home", "Extracurricular_Activities"
        ]
        for key in expected_keys:
            if key not in data:
                return jsonify({"error": f"Missing input: {key}"}), 400

        # Create DataFrame
        df = pd.DataFrame([data])

        # Normalize string inputs
        df["Gender"] = df["Gender"].str.strip().str.capitalize()
        df["Parental_Education_Level"] = df["Parental_Education_Level"].str.strip().str.title()
        df["Internet_Access_at_Home"] = df["Internet_Access_at_Home"].str.strip().str.capitalize()
        df["Extracurricular_Activities"] = df["Extracurricular_Activities"].str.strip().str.capitalize()

        # Convert numeric inputs safely
        df["Study_Hours_per_Week"] = pd.to_numeric(df["Study_Hours_per_Week"], errors="coerce")
        df["Attendance_Rate"] = pd.to_numeric(df["Attendance_Rate"], errors="coerce")
        df["Past_Exam_Scores"] = pd.to_numeric(df["Past_Exam_Scores"], errors="coerce")

        # Check for any invalid or missing numeric inputs
        if df.isnull().values.any():
            return jsonify({"error": "Invalid or missing numeric input"}), 400

        # Debug: Log cleaned input
        print("Cleaned Input:")
        print(df)

        # One-hot encode categorical variables
        df_encoded = pd.get_dummies(df)

        # Align with trained model features
        df_encoded = df_encoded.reindex(columns=feature_columns, fill_value=0)

        # Debug: Log encoded data
        print("Encoded Features:")
        print(df_encoded)

        # Predict
        pred_encoded = model.predict(df_encoded)[0]
        confidence = model.predict_proba(df_encoded)[0].max()
        label = label_encoder.inverse_transform([pred_encoded])[0]

        return jsonify({
            "prediction": label,
            "confidence": round(confidence, 2)
        })

    except Exception as e:
        print("Prediction error:", e)
        return jsonify({"error": "Prediction failed"}), 500

if __name__ == "__main__":
    app.run(debug=True)
