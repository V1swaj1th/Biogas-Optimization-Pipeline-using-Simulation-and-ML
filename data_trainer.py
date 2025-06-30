import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score
import joblib
import os


# Loading Data

data_dir = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data"
training_path = os.path.join(data_dir, "training_data.csv")

df = pd.read_csv(training_path)

# Optional: Basic sanity check
print("🔍 Dataset Summary:\n", df.describe())
assert not df.isnull().values.any(), "Dataset contains missing values!"

# Define features and target
X = df.drop(columns=['CH4_Yield', 'Biogas_Flow'])
y = df['CH4_Yield']

# Step 2: Training Model
model = GradientBoostingRegressor(
    n_estimators=300,
    learning_rate=0.08,
    max_depth=6,
    subsample=0.9,
    random_state=42
)

model.fit(X, y)


# Step 3: Evaluate with CV

rmse_scores = -cross_val_score(model, X, y, cv=5, scoring='neg_root_mean_squared_error')
r2_scores = cross_val_score(model, X, y, cv=5, scoring='r2')

print(f" Mean RMSE over 5-fold CV: {rmse_scores.mean():.4f} L CH4/L")
print(f" Mean R² over 5-fold CV:   {r2_scores.mean():.4f}")


# Step 4: Save Trained Model

model_path = os.path.join(data_dir, "trained_model.joblib")
joblib.dump(model, model_path)
print(f"\n Trained model saved to: {model_path}")
