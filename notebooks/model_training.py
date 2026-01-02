import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# =====================================================
# 🔹 FIXED PATH HANDLING (VERY IMPORTANT)
# =====================================================

# Get project root directory (crop-recommender)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path
DATA_PATH = os.path.join(BASE_DIR, "data", "Crop_recommendation.csv")

# Model save directory
MODEL_DIR = os.path.join(BASE_DIR, "backend", "app")
os.makedirs(MODEL_DIR, exist_ok=True)

# =====================================================
# 🔹 LOAD DATASET
# =====================================================

df = pd.read_csv(DATA_PATH)

# Features & Target
features = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
target = 'label'

X = df[features]
y = df[target]

# =====================================================
# 🔹 LABEL ENCODING
# =====================================================

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# =====================================================
# 🔹 PREPROCESSING + MODEL PIPELINE
# =====================================================

preprocess = ColumnTransformer(
    transformers=[
        ('scale', StandardScaler(), features)
    ]
)

model = Pipeline(steps=[
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(
        n_estimators=400,
        random_state=42
    ))
])

# =====================================================
# 🔹 TRAIN / TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y_encoded
)

# =====================================================
# 🔹 TRAIN MODEL
# =====================================================

model.fit(X_train, y_train)

# =====================================================
# 🔹 PREDICTION & ACCURACY
# =====================================================

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\n🌱 Model Training Completed Successfully!")
print(f"🎯 Accuracy: {accuracy * 100:.2f}%\n")

print("📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# =====================================================
# 🔹 SAVE TRAINED MODEL
# =====================================================

joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))

print("💾 Model saved successfully to:", MODEL_DIR)
