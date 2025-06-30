import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import os

# -------------------------
# Load model
# -------------------------
model_path = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data\trained_model.joblib"
model = joblib.load(model_path)

# -------------------------
# UI: Role selection
# -------------------------
st.set_page_config(page_title="Biogas Dashboard", layout="wide")
st.title("Anaerobic Digestion Monitoring Dashboard")
role = st.sidebar.selectbox("Select Your Role", ["Technician", "Engineer", "Manager", "Executive"])

# -------------------------
# Common form to collect inputs
# -------------------------
st.subheader("Input Parameters")
with st.form("input_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        flow = st.number_input("Flow rate (L/day)", min_value=10.0, max_value=300.0, value=100.0)
        temp1 = st.number_input("Pre-digester Temperature (°C)", min_value=20.0, max_value=60.0, value=35.0)
        temp2 = st.number_input("Main Digester Temperature (°C)", min_value=20.0, max_value=60.0, value=37.0)
        agitator1 = st.number_input("Agitator 1 Power (kW)", min_value=0.0, max_value=5.0, value=1.0)
    with col2:
        agitator2 = st.number_input("Agitator 2 Power (kW)", min_value=0.0, max_value=5.0, value=1.5)
        recycle_ratio = st.slider("Slurry Recycle Ratio (0–1)", 0.0, 1.0, 0.2)
        palm_frac = st.slider("Palm Oil Co-substrate Fraction (0–1)", 0.0, 1.0, 0.25)
        sugar_in = st.number_input("Sugar input (gCOD/L)", min_value=1.0, max_value=10.0, value=5.0)
    with col3:
        vfa = st.number_input("VFA (gCOD/L)", min_value=0.0, max_value=10.0, value=0.05)
        nh3 = st.number_input("NH3 (g N/L)", min_value=0.0, max_value=5.0, value=0.05)
        biogas_flow = st.number_input("Biogas Flow (Nm³/h)", min_value=0.0, max_value=50.0, value=0.0)
    submitted = st.form_submit_button("Predict Methane Yield")

# -------------------------
# Derived Inputs
# -------------------------
VS_in = 25.0
V1 = 300
V2 = 700
total_V = V1 + V2
olr = (flow * VS_in) / total_V
hrt1 = V1 / flow
hrt2 = V2 / flow

# Prepare input DataFrame with all required columns
input_dict = {
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
    'CH4_Yield': 0,  # Placeholder, not used for prediction
    'Biogas_Flow': biogas_flow
}
input_df = pd.DataFrame([input_dict])

# -------------------------
# Prediction & Display
# -------------------------
if submitted:
    # Align input_df columns with model's expected features
    if hasattr(model, 'feature_names_in_'):
        input_df = input_df.reindex(columns=model.feature_names_in_, fill_value=0)
    prediction = model.predict(input_df)[0]
    st.success(f"✅ Predicted Methane Yield: {round(prediction, 4)} L CH₄ / L reactor volume")

    # Optional role-based extras
    if role == "Technician":
        st.metric("Methane Yield (L CH₄/L)", round(prediction, 4))
    elif role == "Engineer":
        st.metric("Methane Yield (L CH₄/L)", round(prediction, 4))
        st.subheader("Engineer View: Input vs Yield")
        fig, ax = plt.subplots()
        sns.scatterplot(x=[flow], y=[prediction], ax=ax)
        ax.set_xlabel("Flowrate")
        ax.set_ylabel("Methane Yield")
        ax.set_title("Flowrate vs Methane Yield")
        st.pyplot(fig)
    elif role == "Manager":
        st.metric("Methane Yield (L CH₄/L)", round(prediction, 4))
        st.info("Manager: High Recycle Ratios or VFA can reduce yield.")
        st.write("⚠️ Check recycle ratio or VFA concentration if yield is low.")
    elif role == "Executive":
        st.metric("Methane Yield (L CH₄/L)", round(prediction, 4))
        st.caption("📊 Summary: Predicted CH₄ yield based on current plant settings.")

# -------------------------
# Footer
# -------------------------
st.caption("Built using Streamlit | IIT Indore Biogas Project")
