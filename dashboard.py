import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
import os
from predictive_model import DesalinationPredictor
from optimizer import PlantOptimizer

# --- Page Configuration ---
st.set_page_config(
    page_title="AQUA-IA Digital Twin POC",
    page_icon="💧",
    layout="wide"
)

st.title("💧 AQUA-IA: Desalination Plant Digital Twin & Optimizer")
st.markdown("An AI-powered Proof of Concept for intelligent plant management.")

# --- Initialize Models and Optimizer ---
@st.cache_resource
def load_components():
    """Loads or trains the predictive model and initializes the optimizer."""
    predictor = DesalinationPredictor()
    
    # Check if models exist, if not, train them
    if not (os.path.exists('energy_model.pkl') and os.path.exists('quality_model.pkl') and os.path.exists('fouling_model.pkl')):
        with st.spinner("Training AI models for the first time... This may take a minute."):
            from data_generator import generate_desalination_data
            data = generate_desalination_data()
            data.to_csv('desal_data.csv', index=False)
            predictor.train('desal_data.csv')
    else:
        predictor.load_models()
        
    optimizer = PlantOptimizer(predictor)
    return predictor, optimizer

predictor, optimizer = load_components()

# --- Sidebar for User Inputs ---
st.sidebar.header("1. Current Plant Settings & Conditions")

# User-adjustable inputs
current_pressure = st.sidebar.slider("Feed Pressure (bar)", 60.0, 85.0, 72.5)
current_flow = st.sidebar.slider("Feed Flow (m³/h)", 800.0, 1200.0, 1000.0)
current_dosage = st.sidebar.slider("Chemical Dosage (mg/L)", 1.0, 3.0, 2.0)

# Environmental conditions (fixed for a given optimization run)
current_temp = st.sidebar.slider("Seawater Temperature (°C)", 20.0, 35.0, 28.0)
current_salinity = st.sidebar.slider("Feed Salinity (ppm)", 38000, 45000, 41000)

current_conditions = {
    'seawater_temp': current_temp,
    'feed_salinity': current_salinity
}

# --- Main Dashboard Area ---

# Section 1: Predictive Performance
st.header("🔮 Predictive Performance Analysis (Digital Twin)")

current_input_df = pd.DataFrame([{
    'feed_pressure': current_pressure,
    'feed_flow': current_flow,
    'chemical_dosage': current_dosage,
    'seawater_temp': current_temp,
    'feed_salinity': current_salinity
}])

pred_energy, pred_quality, pred_fouling = predictor.predict(current_input_df)

col1, col2, col3 = st.columns(3)
col1.metric("Predicted Energy (kWh/m³)", f"{pred_energy:.3f}")
col2.metric("Predicted Water TDS (ppm)", f"{pred_quality:.1f}")
col3.metric("Predicted Fouling Rate", f"{pred_fouling:.5f}")

# Predictive Maintenance Alert
if pred_fouling > 0.006:
    st.warning("⚠️ **Predictive Maintenance Alert:** High fouling rate detected. Membrane lifespan may be significantly reduced. Consider increasing chemical dosage or reviewing operating parameters.")

# Section 2: Optimization
st.header("🚀 AI-Powered Optimization")

# Use a form to prevent re-running on every widget change
with st.form("optimization_form"):
    submitted = st.form_submit_button("Find Optimal Settings", type="primary")
    if submitted:
        with st.spinner("AI is analyzing the system to find the most cost-effective strategy..."):
            # We need to set the current conditions for the optimizer to calculate the baseline cost
            optimizer.current_seawater_temp = current_temp
            optimizer.current_feed_salinity = current_salinity
            current_cost = optimizer.objective_function([current_pressure, current_flow, current_dosage])
            
            # Run the optimization
            optimization_results = optimizer.optimize(current_conditions)

        if optimization_results['success']:
            st.success("✅ Optimization Complete! Found a more efficient operating point.")
            
            # Display results side-by-side
            col_opt1, col_opt2 = st.columns(2)
            
            with col_opt1:
                st.subheader("Current Settings")
                st.write(f"**Pressure:** {current_pressure:.2f} bar")
                st.write(f"**Flow:** {current_flow:.0f} m³/h")
                st.write(f"**Dosage:** {current_dosage:.2f} mg/L")
                st.metric("Estimated Hourly Cost", f"${current_cost:.2f}")

            with col_opt2:
                st.subheader("AI-Recommended Settings")
                st.write(f"**Pressure:** {optimization_results['optimal_pressure']:.2f} bar")
                st.write(f"**Flow:** {optimization_results['optimal_flow']:.0f} m³/h")
                st.write(f"**Dosage:** {optimization_results['optimal_dosage']:.2f} mg/L")
                st.metric("Optimized Hourly Cost", f"${optimization_results['optimal_cost_per_hour']:.2f}", 
                           f"{-(current_cost - optimization_results['optimal_cost_per_hour'])/current_cost*100:.2f}%")

            # Detailed comparison
            st.subheader("Detailed Performance Comparison")
            comparison_data = {
                'Metric': ['Energy (kWh/m³)', 'Water Quality (TDS ppm)', 'Fouling Rate'],
                'Current': [pred_energy, pred_quality, pred_fouling],
                'Optimized': [optimization_results['predicted_energy'], optimization_results['predicted_quality'], optimization_results['predicted_fouling']]
            }
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True)

        else:
            st.error("❌ Optimization failed. Please check the input parameters or constraints.")


# Section 3: Model Insights
st.header("📊 Model Insights")
st.write("This chart shows how the AI model understands the relationship between pressure and energy consumption, holding other factors constant.")

# Generate data for a sensitivity chart
pressure_range = np.linspace(60, 85, 50)
sensitivity_data = []
for p in pressure_range:
    df_sens = pd.DataFrame([{
        'feed_pressure': p,
        'feed_flow': 1000,
        'chemical_dosage': 2.0,
        'seawater_temp': 28.0,
        'feed_salinity': 41000
    }])
    e, q, f = predictor.predict(df_sens)
    sensitivity_data.append({'Pressure': p, 'Energy': e})

df_sensitivity = pd.DataFrame(sensitivity_data)
fig = px.line(df_sensitivity, x='Pressure', y='Energy', title='Energy Consumption vs. Feed Pressure')
st.plotly_chart(fig, use_container_width=True)
