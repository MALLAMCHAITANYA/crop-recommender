import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load Kaggle dataset
DATA_PATH = os.path.join("..", "data", "Crop_recommendation.csv")
df = pd.read_csv(DATA_PATH)

# Features & Target
features = ['N','P','K','temperature','humidity','ph','rainfall']
target = 'label'

X = df[features]
y = df[target]

# Encode labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
MODEL_DIR = os.path.join("..", "backend", "app")
os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(label_encoder, os.path.join(MODEL_DIR, "label_encoder.pkl"))

# Preprocessing
preprocess = ColumnTransformer([
    ('scale', StandardScaler(), features)
])

# Model
model = Pipeline([
    ('preprocess', preprocess),
    ('classifier', RandomForestClassifier(n_estimators=400, random_state=42))
])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Train
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Results
accuracy = accuracy_score(y_test, y_pred)
print(f"\n🌱 Model Training Completed Successfully!")
print(f"🎯 Accuracy: {accuracy*100:.2f}%\n")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
print("💾 Model saved to:", MODEL_DIR)
