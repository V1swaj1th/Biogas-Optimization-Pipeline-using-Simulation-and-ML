# Biogas-Optimization-Pipeline-using-Simulation-and-ML
A data-driven modeling pipeline to optimize methane yield in anaerobic digesters. Includes a 2-stage ADM1-inspired simulator with physiological penalties, synthetic dataset generation, ML-based methane prediction, feature importance analysis, and a multi-role Streamlit dashboard for monitoring and decision support.

This project simulates and optimizes methane production in a two-stage anaerobic digester using a custom-built mathematical model + machine learning. The idea was to create a reliable digital approximation of a real biogas plant, especially since I didn’t have real sensor data.

I simulate various process parameters (like temperature, OLR, recycle ratio, agitator power etc.), generate synthetic but realistic training data, and train an ML model (Gradient Boosting) to predict methane yield. To make sure it’s not just curve-fitting, I also applied physiological penalties (e.g., NH₃ and VFA inhibition, temperature deviation, poor mixing, etc.) so the outputs reflect realistic trends.

This project includes the following - 
A 2-stage ADM1-inspired model that simulates hydrolysis to methanogenesis using ODEs

Penalties for conditions that affect microbial activity (ammonia, temperature, mixing, etc.)

Code to generate synthetic data over a wide range of process variables

An ML pipeline (Gradient Boosting Regressor) trained on this simulated data

Feature importance analysis + graphs to visualize how variables affect CH₄ yield

A Streamlit dashboard with 4 access levels (Technician, Engineer, Manager, Executive)

Role-based monitoring, prediction, alerts, and visualizations
