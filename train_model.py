import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib
import os

# Ensure directory exists
os.makedirs("trained_data", exist_ok=True)

# Load dataset
df = pd.read_csv("csv/student_performance_dataset.csv")

# Drop ID
df = df.drop(columns=["Student_ID"])

# Encode target
label_encoder = LabelEncoder()
df["Pass_Fail_Label"] = label_encoder.fit_transform(df["Pass_Fail"])
joblib.dump(label_encoder, "trained_data/pass_fail_encoder.pkl")

# Features and target
y = df["Pass_Fail_Label"]
X = df.drop(columns=["Pass_Fail", "Pass_Fail_Label", "Final_Exam_Score"])
X = pd.get_dummies(X)
joblib.dump(X.columns.tolist(), "trained_data/model_features.pkl")

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model pipeline
clf_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42))
])

# Train and save
clf_pipeline.fit(X_train, y_train)
print("Test Accuracy:", clf_pipeline.score(X_test, y_test))
joblib.dump(clf_pipeline, "trained_data/student_pass_fail_model.pkl")
print("✅ Model training done.")
