import numpy as np
import pandas as pd
from scipy.integrate import solve_ivp
import os

# Constants
V_stage1 = 300  # L
V_stage2 = 700  # L
VS_in = 25.0    # gVS/L
R = 8.314
T_ref = 308.15  # 35°C in K

# Parameters
params = {
    'k_hyd': 1.0, 'K_hyd': 0.3, 'Ea_hyd': 60000, 'b_hyd': 0.05,
    'k_aco': 5.0, 'K_aco': 0.3, 'Y_aco': 0.1, 'Ea_aco': 65000, 'b_aco': 0.05,
    'k_ace': 4.0, 'K_ace': 0.2, 'Y_ace': 0.1, 'Ea_ace': 70000, 'b_ace': 0.03,
    'k_meth': 6.0, 'K_meth': 0.1, 'Y_meth': 0.1, 'Ea_meth': 75000, 'b_meth': 0.02,
    'KI_vfa': 1.0, 'KI_nh3': 0.5
}

# Arrhenius equation
def arrhenius(k_ref, Ea, T):
    return k_ref * np.exp(-Ea / R * (1/T - 1/T_ref))

# Penalties
def mixing_efficiency(power_kw, stage_vol):
    return np.clip(np.log1p(power_kw) / np.log1p(stage_vol / 100), 0.2, 1.0)

def temperature_penalty(T):
    T_C = T - 273.15
    return np.clip(1 - 0.01 * (T_C - 36)**2, 0.5, 1.0)

def nh3_penalty(NH3):
    return 1 / (1 + (NH3 / 0.5)**2)

# ADM1 based modelling
def adm1_stage(t, state, params, Q_in, S_su_in, V_reactor, T, agitator_kw, recycle_ratio):
    S_su, X_hyd, S_aa, X_aco, S_ac, X_ace, S_ch4, S_vfa, S_nh3, X_meth = state

    mix_eff = mixing_efficiency(agitator_kw, V_reactor)
    T_pen = temperature_penalty(T)
    nh3_pen = nh3_penalty(S_nh3)

    k_hyd = arrhenius(params['k_hyd'], params['Ea_hyd'], T) * mix_eff * T_pen
    k_aco = arrhenius(params['k_aco'], params['Ea_aco'], T) * mix_eff * T_pen
    k_ace = arrhenius(params['k_ace'], params['Ea_ace'], T) * mix_eff * T_pen
    k_meth = arrhenius(params['k_meth'], params['Ea_meth'], T) * mix_eff * T_pen

    mu_hyd = (k_hyd * S_su) / (params['K_hyd'] + S_su + 1e-6)
    mu_aco = (k_aco * S_su) / (params['K_aco'] + S_su + 1e-6)
    mu_ace = (k_ace * S_aa) / (params['K_ace'] + S_aa + 1e-6)
    mu_meth = (k_meth * S_ac) / (params['K_meth'] + S_ac + 1e-6)

    inh_vfa = 1 / (1 + (S_vfa / params['KI_vfa'])**2)
    inh_nh3 = 1 / (1 + (S_nh3 / params['KI_nh3'])**2)
    mu_meth *= inh_vfa * inh_nh3 * nh3_pen

    r_hyd = mu_hyd * X_hyd
    r_aco = mu_aco * X_aco
    r_ace = mu_ace * X_ace
    r_meth = mu_meth * X_meth

    r_su_used = r_aco / params['Y_aco']
    r_aa_used = r_ace / params['Y_ace']
    r_ac_used = r_meth / params['Y_meth']

    dS_su = -r_su_used + r_hyd + (Q_in/V_reactor)*(S_su_in - S_su)
    dX_hyd = r_hyd - params['b_hyd'] * X_hyd
    dS_aa = -r_aa_used + r_aco - (Q_in/V_reactor)*S_aa
    dX_aco = r_aco - params['b_aco'] * X_aco
    dS_ac = -r_ac_used + r_ace - (Q_in/V_reactor)*S_ac
    dX_ace = r_ace - params['b_ace'] * X_ace
    dS_ch4 = r_meth * 0.35
    dS_vfa = r_aco * 0.5 - r_ace * 0.3 - (Q_in/V_reactor)*S_vfa
    dS_nh3 = r_aco * 0.2 - (Q_in/V_reactor)*S_nh3
    dX_meth = r_meth - params['b_meth'] * X_meth

    return [dS_su, dX_hyd, dS_aa, dX_aco, dS_ac, dX_ace, dS_ch4, dS_vfa, dS_nh3, dX_meth]

# Taking 2-stage AD system into consideration for simulations
def simulate_two_stage_system(Q_in, T1_C, T2_C, agitator1_kw, agitator2_kw, recycle_ratio, palm_frac, S_su_in, hours=30):
    t_span = (0, hours)
    t_eval = np.linspace(*t_span, 300)
    T1 = T1_C + 273.15
    T2 = T2_C + 273.15
    sugar_adj = S_su_in * (1 - palm_frac)

    init_state = [sugar_adj, 0.02, 0.05, 0.01, 0.1, 0.005, 0.0, 0.03, 0.03, 0.005]

    try:
        res1 = solve_ivp(
            adm1_stage, t_span, init_state, args=(params, Q_in, sugar_adj, V_stage1, T1, agitator1_kw, recycle_ratio),
            t_eval=t_eval, method="LSODA", rtol=1e-5, atol=1e-6
        )
        outlet_stage1 = np.clip(res1.y[:, -1], 0, np.inf)
    except Exception as e:
        return {'CH4_Yield': np.nan}

    try:
        res2 = solve_ivp(
            adm1_stage, t_span, outlet_stage1, args=(params, Q_in, outlet_stage1[0], V_stage2, T2, agitator2_kw, recycle_ratio),
            t_eval=t_eval, method="LSODA", rtol=1e-5, atol=1e-6
        )
        outlet_stage2 = np.clip(res2.y[:, -1], 0, np.inf)
    except Exception as e:
        return {'CH4_Yield': np.nan}

    ch4 = outlet_stage2[6]
    biogas = ch4 * (Q_in / 1000) * 0.65 / (hours / 24)

    return {
        'CH4_Yield': ch4,
        'Biogas_Flow': biogas,
        'Final_VFA': outlet_stage2[7],
        'Final_NH3': outlet_stage2[8]
    }

# Dataset generation
flows = np.linspace(30, 180, 6)
temps = [(28, 32), (32, 36), (36, 40), (40, 45)]
agitator1_powers = np.linspace(0.2, 2.0, 5)
agitator2_powers = np.linspace(0.5, 3.0, 5)
recycle_ratios = np.linspace(0.05, 0.5, 5)
palm_fracs = np.linspace(0.0, 0.5, 6)
sugars = np.linspace(2.0, 10.0, 5)

records = []
for Q in flows:
    for (T1, T2) in temps:
        for A1 in agitator1_powers:
            for A2 in agitator2_powers:
                for R in recycle_ratios:
                    for PF in palm_fracs:
                        for S in sugars:
                            result = simulate_two_stage_system(Q, T1, T2, A1, A2, R, PF, S)
                            if not np.isfinite(result['CH4_Yield']):
                                continue
                            records.append({
                                'FlowRate': Q,
                                'Temp1': T1,
                                'Temp2': T2,
                                'Agitator1_kW': A1,
                                'Agitator2_kW': A2,
                                'Recycle_Ratio': R,
                                'PalmFrac': PF,
                                'SugarIn': S,
                                'OLR': (Q * VS_in) / (V_stage1 + V_stage2),
                                'HRT1': V_stage1 / Q,
                                'HRT2': V_stage2 / Q,
                                'VFA': result['Final_VFA'],
                                'NH3': result['Final_NH3'],
                                'CH4_Yield': result['CH4_Yield'],
                                'Biogas_Flow': result['Biogas_Flow']
                            })

df = pd.DataFrame(records)
output_dir = r"C:\Users\viswa\Desktop\IITI\Digestor Project\importanceMODEL\data"
os.makedirs(output_dir, exist_ok=True)
df.to_csv(os.path.join(output_dir, "training_data.csv"), index=False)
df.sample(frac=0.2, random_state=42).to_csv(os.path.join(output_dir, "validation_data.csv"), index=False)
print(" Data generated and saved.")
