import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import numpy as np
# Load dataset
df = pd.read_csv('fossil_fuel_emissions.csv')

# Feature selection
features = [
    'oil_production_barrels',
    'gas_production_mcf',
    'coal_production_tons',
    'flared_gas_mcf',
    'carbon_capture_tons',
    'year'  # Year is included to help the model learn temporal trends
]

target = 'scope1_emissions_tons'

X = df[features]
y = df[target]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost Regressor
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=200, learning_rate=0.1)
model.fit(X_train, y_train)

# Evaluate model
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))

r2 = r2_score(y_test, y_pred)

print(f"✅ Model trained successfully!")
print(f"📊 RMSE: {rmse:.2f}")
print(f"📈 R² Score: {r2:.4f}")

# Save the model
joblib.dump(model, 'model.pkl')
print("📁 Model saved as model.pkl")
