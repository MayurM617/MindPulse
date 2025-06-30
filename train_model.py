import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

# Load dataset
df = pd.read_csv('emotions.csv')

# Drop rows with missing values
df.dropna(inplace=True)

# Filter out extreme outliers
q_low = df.drop(columns=['label']).quantile(0.01)
q_high = df.drop(columns=['label']).quantile(0.99)
df = df[(df.drop(columns=['label']) >= q_low).all(axis=1) & (df.drop(columns=['label']) <= q_high).all(axis=1)]

# Features & target
X = df.drop(columns=['label'])
y = df['label']

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Save label encoder
joblib.dump(label_encoder, 'label_encoder.pkl')

# Normalize feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save scaler
joblib.dump(scaler, 'scaler.pkl')

# Save features list
with open('features.json', 'w') as f:
    json.dump(list(X.columns), f)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.55, random_state=310)

# Train XGBoost model
model = xgb.XGBClassifier(objective='multi:softmax', num_class=len(label_encoder.classes_), eval_metric='mlogloss')
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Save model
joblib.dump(model, 'model.pkl')

print("✅ Training complete. Model and preprocessors saved.")
