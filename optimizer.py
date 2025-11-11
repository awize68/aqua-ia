import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution
from predictive_model import DesalinationPredictor

class PlantOptimizer:
    def __init__(self, predictor: DesalinationPredictor):
        print("DEBUG: __init__ of PlantOptimizer is starting...")
        self.predictor = predictor
        
        self.energy_cost_per_kwh = 0.15
        self.chemical_cost_per_mg_l = 0.002
        print(f"DEBUG: chemical_cost_per_mg_l set to: {self.chemical_cost_per_mg_l}")
        self.membrane_replacement_cost_per_index_point = 50000
        
        self.max_tds_ppm = 150
        self.min_flow_m3_h = 900
        print("DEBUG: __init__ of PlantOptimizer finished.")

    def objective_function(self, params):
        pressure, flow, dosage = params
        
        input_data = pd.DataFrame([{
            'feed_pressure': pressure,
            'feed_flow': flow,
            'chemical_dosage': dosage,
            'seawater_temp': self.current_seawater_temp,
            'feed_salinity': self.current_feed_salinity
        }])
        
        energy, quality, fouling = self.predictor.predict(input_data)
        
        energy_cost = energy * flow * self.energy_cost_per_kwh
        chemical_cost = dosage * flow * self.chemical_cost_per_mg_l
        fouling_cost = fouling * self.membrane_replacement_cost_per_index_point
        total_cost = energy_cost + chemical_cost + fouling_cost
        
        penalty = 0
        if quality > self.max_tds_ppm:
            penalty += 1e6
        if flow < self.min_flow_m3_h:
            penalty += 1e6
            
        return total_cost + penalty

    def optimize(self, current_conditions):
        self.current_seawater_temp = current_conditions['seawater_temp']
        self.current_feed_salinity = current_conditions['feed_salinity']
        
        bounds = [(60, 85), (800, 1200), (1.0, 3.0)]
        
        print("Running optimization...")
        result = differential_evolution(
            self.objective_function, 
            bounds, 
            strategy='best1bin', 
            maxiter=25, 
            popsize=15, 
            tol=0.01, 
            mutation=(0.5, 1), 
            recombination=0.7,
            seed=42
        )
        
        if result.success:
            optimal_params = result.x
            optimal_cost = result.fun
            
            input_data = pd.DataFrame([{
                'feed_pressure': optimal_params[0],
                'feed_flow': optimal_params[1],
                'chemical_dosage': optimal_params[2],
                'seawater_temp': self.current_seawater_temp,
                'feed_salinity': self.current_feed_salinity
            }])
            opt_energy, opt_quality, opt_fouling = self.predictor.predict(input_data)
            
            return {
                "success": True,
                "optimal_pressure": optimal_params[0],
                "optimal_flow": optimal_params[1],
                "optimal_dosage": optimal_params[2],
                "optimal_cost_per_hour": optimal_cost,
                "predicted_energy": opt_energy,
                "predicted_quality": opt_quality,
                "predicted_fouling": opt_fouling
            }
        else:
            return {"success": False, "message": result.message}
