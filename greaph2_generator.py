import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os


# Load Trained Model & Data

model_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\trained_model.joblib"
data_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\training_data.csv"

model = joblib.load(model_path)
df = pd.read_csv(data_path)


# Density Contour Plot with Colorbar


def plot_density_contour_with_colorbar(x, y, xlabel, ylabel, title, cmap='viridis'):
    plt.figure(figsize=(10, 6))
    ax = sns.kdeplot(
        data=df,
        x=x,
        y=y,
        fill=True,
        cmap=cmap,
        cbar=True,
        cbar_kws={'label': 'Density'}
    )
    sns.scatterplot(data=df, x=x, y=y, color='white', s=5, alpha=0.3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# Generate All Requested Graphs


# 1. Temperature vs Methane Yield (Main digester Temp2)
plot_density_contour_with_colorbar(
    x='Temp2',
    y='CH4_Yield',
    xlabel='Temperature Stage 2 (°C)',
    ylabel='Methane Yield (L CH₄/L)',
    title='Temperature (Stage 2) vs Methane Yield'
)

# 2. OLR vs Methane Yield
plot_density_contour_with_colorbar(
    x='OLR',
    y='CH4_Yield',
    xlabel='Organic Loading Rate (gVS/L/day)',
    ylabel='Methane Yield (L CH₄/L)',
    title='OLR vs Methane Yield'
)

# 3. HRT2 vs Methane Yield
plot_density_contour_with_colorbar(
    x='HRT2',
    y='CH4_Yield',
    xlabel='HRT Stage 2 (days)',
    ylabel='Methane Yield (L CH₄/L)',
    title='Retention Time (Stage 2) vs Methane Yield'
)

# 4. Biogas Flow vs Methane Yield
plot_density_contour_with_colorbar(
    x='Biogas_Flow',
    y='CH4_Yield',
    xlabel='Biogas Flow (Nm³/h)',
    ylabel='Methane Yield (L CH₄/L)',
    title='Biogas Flow vs Methane Yield'
)

# 5. Palm Fraction vs Methane Yield
plot_density_contour_with_colorbar(
    x='PalmFrac',
    y='CH4_Yield',
    xlabel='Palm Oil Co-substrate Fraction',
    ylabel='Methane Yield (L CH₄/L)',
    title='Palm Oil Fraction vs Methane Yield'
)

# 6. Recycle Ratio vs Methane Yield
plot_density_contour_with_colorbar(
    x='Recycle_Ratio',
    y='CH4_Yield',
    xlabel='Recycle Ratio',
    ylabel='Methane Yield (L CH₄/L)',
    title='Recycle Ratio vs Methane Yield'
)
