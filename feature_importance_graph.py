import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score

# Loadiong the model and data

model_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\trained_model.joblib"
data_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\training_data.csv"

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Split features and target
X = df.drop(columns=["CH4_Yield", "Biogas_Flow"]) 
y = df["CH4_Yield"]

# Baseline prediction for reference
y_pred = model.predict(X)
print(f"Baseline R² score: {r2_score(y, y_pred):.4f}")

# Compute permutation importance
result = permutation_importance(model, X, y, n_repeats=30, random_state=42, scoring='r2')

# Create importance dataframe
importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': result.importances_mean,
    'Std': result.importances_std
}).sort_values(by="Importance", ascending=False)

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=importance_df, x="Importance", y="Feature", palette="crest", edgecolor='black')
plt.title("Permutation Feature Importance (Impact on Methane Yield)")
plt.xlabel("Mean Decrease in R² Score")
plt.grid(True, axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
