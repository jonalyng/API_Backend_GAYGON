# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
import joblib
import os

# 📁 Ensure directory exists for saving model
os.makedirs("trained_data", exist_ok=True)

# 📥 Load dataset
df = pd.read_csv("csv/student_performance_dataset.csv")

# 🧹 Drop ID column — not needed for training
df = df.drop(columns=["Student_ID"])

# 🔤 Encode target variable Pass/Fail
label_encoder = LabelEncoder()
df["Pass_Fail_Label"] = label_encoder.fit_transform(df["Pass_Fail"])  # Pass=1, Fail=0
joblib.dump(label_encoder, "trained_data/pass_fail_encoder.pkl")

# 🎯 Set target and features
y = df["Pass_Fail_Label"]
X = df.drop(columns=["Pass_Fail", "Pass_Fail_Label", "Final_Exam_Score"])  # Optionally drop Final_Exam_Score

# 🔄 One-hot encode categorical features
X = pd.get_dummies(X)

# 💾 Save feature columns for later use
joblib.dump(X.columns.tolist(), "trained_data/model_features.pkl")

# 📊 Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Create pipeline
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", MLPClassifier(
        hidden_layer_sizes=(64,),
        activation="relu",
        max_iter=2000,
        early_stopping=True,
        random_state=42
    ))
])

# 🏋️‍♂️ Train model
clf_pipeline.fit(X_train, y_train)

# 💾 Save trained model
joblib.dump(clf_pipeline, "trained_data/student_pass_fail_model.pkl")

print("✅ Training complete. Model saved in 'trained_data/'")
