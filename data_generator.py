import pandas as pd
import numpy as np

def generate_desalination_data(num_samples=10000):
    """
    Generates realistic, correlated data for a seawater desalination plant.
    This simulates the complex relationships between inputs and outputs.
    """
    print("Generating simulated plant data...")
    
    # Base operating parameters with some randomness
    feed_pressure = np.random.uniform(60, 85, num_samples)  # bar
    feed_flow = np.random.uniform(800, 1200, num_samples)    # m3/h
    chemical_dosage = np.random.uniform(1.0, 3.0, num_samples) # mg/L
    
    # Environmental factors (simulating daily/seasonal changes)
    seawater_temp = np.random.uniform(20, 35, num_samples)   # Celsius
    feed_salinity = np.random.uniform(38000, 45000, num_samples) # ppm (TDS)

    # --- Simulate Core Process Relationships (The "Physics" of the plant) ---
    
    # 1. Energy Consumption (kWh/m3)
    # Increases with pressure and flow. Decreases slightly with temperature (warmer water is easier to process).
    base_energy = (feed_pressure / 70.0) * 3.5
    energy_consumption = base_energy * (1 + (feed_flow - 1000) / 2000) * (1 - (seawater_temp - 27.5) * 0.01)
    energy_consumption += np.random.normal(0, 0.1, num_samples) # Add noise

    # 2. Water Quality (TDS in ppm)
    # Improves with higher pressure, worsens with higher feed salinity and temperature.
    base_tds = feed_salinity / 1000
    water_quality = base_tds * (1 - (feed_pressure - 72.5) * 0.02) * (1 + (seawater_temp - 27.5) * 0.005)
    water_quality += np.random.normal(0, 5, num_samples) # Add noise
    water_quality = np.clip(water_quality, 50, 500) # Ensure realistic range

    # 3. Membrane Fouling Rate (Index/hour)
    # This is the CRITICAL predictive element.
    # Increases with temperature and salinity. Decreases with chemical dosage.
    # High pressure can also accelerate fouling by compacting matter onto the membrane.
    fouling_rate = (seawater_temp / 30.0) * (feed_salinity / 41000.0) * (1 - chemical_dosage / 4.0)
    fouling_rate *= (1 + (feed_pressure - 72.5) * 0.01) # Pressure effect
    fouling_rate *= (1 + (feed_flow - 1000) / 5000)      # Flow effect
    fouling_rate += np.random.normal(0, 0.001, num_samples)
    fouling_rate = np.clip(fouling_rate, 0.0001, 0.01) # Ensure positive and realistic range

    # Assemble into a DataFrame
    data = pd.DataFrame({
        'feed_pressure': feed_pressure,
        'feed_flow': feed_flow,
        'chemical_dosage': chemical_dosage,
        'seawater_temp': seawater_temp,
        'feed_salinity': feed_salinity,
        'energy_kwh_per_m3': energy_consumption,
        'product_tds_ppm': water_quality,
        'fouling_rate_index_per_hr': fouling_rate
    })
    
    print("Data generation complete.")
    return data

if __name__ == '__main__':
    df = generate_desalination_data()
    print(df.head())
    df.to_csv('desal_data.csv', index=False)