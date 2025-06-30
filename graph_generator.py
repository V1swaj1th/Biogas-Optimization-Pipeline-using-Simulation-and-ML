import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.inspection import permutation_importance

# loading model and data
model_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\trained_model.joblib"
data_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\training_data.csv"

model = joblib.load(model_path)
df = pd.read_csv(data_path)

# Features and target
X = df.drop(columns=['CH4_Yield', 'Biogas_Flow'])
y = df["CH4_Yield"]
y_pred = model.predict(X)

# 1. Methane Yield vs Temperature (Temp2)
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Temp2"], y=y_pred)
plt.title("Methane Yield vs Main Digester Temperature")
plt.xlabel("Temperature (°C)")
plt.ylabel("Predicted CH₄ Yield (L/L)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 2. Methane Yield vs OLR
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["OLR"], y=y_pred)
plt.title("Methane Yield vs OLR")
plt.xlabel("OLR (gVS/L/day)")
plt.ylabel("Predicted CH₄ Yield (L/L)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 3. Methane Yield vs HRT2
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["HRT2"], y=y_pred)
plt.title("Methane Yield vs Main Digester HRT")
plt.xlabel("HRT2 (days)")
plt.ylabel("Predicted CH₄ Yield (L/L)")
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. Contour Plot: Temp2 vs PalmFrac
pivot_df = df.copy()
pivot_df["Predicted_CH4"] = y_pred
pivot_table = pivot_df.pivot_table(values="Predicted_CH4", index="PalmFrac", columns="Temp2")
plt.figure(figsize=(10, 6))
sns.heatmap(pivot_table, cmap="viridis", annot=True, fmt=".2f")
plt.title("CH₄ Yield (L/L) vs Temp2 and Palm Oil Fraction")
plt.xlabel("Temperature (°C)")
plt.ylabel("Palm Oil Fraction")
plt.tight_layout()
plt.show()

# 5. Feature Importance
importances = model.feature_importances_
features = X.columns
sorted_idx = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], importances[sorted_idx], color="steelblue")
plt.xlabel("Relative Importance")
plt.title("Feature Importance (Gradient Boosting)")
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 6. Actual vs Predicted
plt.figure(figsize=(8, 6))
plt.scatter(y, y_pred, alpha=0.6)
plt.plot([min(y), max(y)], [min(y), max(y)], 'r--')
plt.xlabel("Actual CH₄ Yield (L/L)")
plt.ylabel("Predicted CH₄ Yield (L/L)")
plt.title("Predicted vs Actual Methane Yield")
plt.grid(True)
plt.tight_layout()
plt.show()
