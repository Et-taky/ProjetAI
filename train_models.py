# train_models.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import joblib
import os

# téléchargement du dataset
df = pd.read_csv("data.csv")  

# élimination des colonnes unutiles
df.drop(columns=["id", "Unnamed: 32"], inplace=True)

# Encode diagnosis (B = 0, M = 1)
df["diagnosis"] = df["diagnosis"].map({'B': 0, 'M': 1})

# Separate features and target
X = df[['concave points_mean','concave points_worst','texture_mean','radius_worst','perimeter_worst','fractal_dimension_se','texture_worst','concave points_se','area_se','smoothness_worst','concavity_se']]
y = df["diagnosis"]

# Save feature names for the Flask app
feature_names = ['concave points_mean','concave points_worst','texture_mean','radius_worst','perimeter_worst','fractal_dimension_se','texture_worst','concave points_se','area_se','smoothness_worst','concavity_se']
print(feature_names )

# Normalize data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Initialize models
log_reg = LogisticRegression()
knn = KNeighborsClassifier()
tree = DecisionTreeClassifier(random_state=42)

# Fit models
log_reg.fit(X_train, y_train)
knn.fit(X_train, y_train)
tree.fit(X_train, y_train)

# Create 'models' folder if it doesn't exist
os.makedirs("models", exist_ok=True)

# Save models and scaler
joblib.dump(log_reg, 'models/log_reg.pkl')
joblib.dump(knn, 'models/knn.pkl')
joblib.dump(tree, 'models/tree.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
joblib.dump(feature_names, 'models/feature_names.pkl')

print("✅ All models and scaler saved successfully.")
