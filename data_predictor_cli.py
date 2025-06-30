import pandas as pd
import joblib
import os


# Load Trained Model

model_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\trained_model.joblib"
model = joblib.load(model_path)


# Predict from CLI Input

def predict_methane_yield_cli():
    print("\n--- Enter Input Parameters for Methane Prediction ---")
    try:
        flow = float(input("Flow rate (L/day) [50–150]: "))
        temp1 = float(input("Pre-digester Temperature (°C) [30–40]: "))
        temp2 = float(input("Main Digester Temperature (°C) [35–42]: "))
        agitator1 = float(input("Agitator 1 Power (kW) [0.5–2.0]: "))
        agitator2 = float(input("Agitator 2 Power (kW) [1.0–3.0]: "))
        recycle_ratio = float(input("Slurry Recycle Ratio (0–1) [0.1–0.5]: "))
        palm_frac = float(input("Palm Oil Co-substrate Fraction (0–1) [0.0–0.5]: "))
        sugar_in = float(input("Sugar input (gCOD/L) [3.0–8.0]: "))
        vfa = float(input("VFA (gCOD/L) [default 0.05]: ") or 0.05)
        nh3 = float(input("NH3 (g N/L) [default 0.05]: ") or 0.05)
        biogas_flow = float(input("Biogas Flow (Nm³/h) [default 0.0 if unknown]: ") or 0.0)

    except ValueError:
        print("❌ Invalid input. Please enter numeric values.")
        return

    # Derived parameters
    VS_in = 25.0
    V1 = 300
    V2 = 700
    total_V = V1 + V2
    olr = (flow * VS_in) / total_V
    hrt1 = V1 / flow
    hrt2 = V2 / flow

    input_df = pd.DataFrame([{
        'FlowRate': flow,
        'Temp1': temp1,
        'Temp2': temp2,
        'Agitator1_kW': agitator1,
        'Agitator2_kW': agitator2,
        'Recycle_Ratio': recycle_ratio,
        'PalmFrac': palm_frac,
        'SugarIn': sugar_in,
        'OLR': olr,
        'HRT1': hrt1,
        'HRT2': hrt2,
        'VFA': vfa,
        'NH3': nh3,
        'Biogas_Flow': biogas_flow
    }])

    prediction = model.predict(input_df)[0]
    print(f"\n✅ Predicted Methane Yield: {round(prediction, 4)} L CH₄ / L reactor volume")


# Run CLI

if __name__ == "__main__":
    predict_methane_yield_cli()
