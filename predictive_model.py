import pandas as pd
import lightgbm as lgb
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

class DesalinationPredictor:
    """
    The "Digital Twin". A set of AI models trained to predict the plant's behavior.
    """
    def __init__(self):
        self.energy_model = lgb.LGBMRegressor(random_state=42)
        self.quality_model = lgb.LGBMRegressor(random_state=42)
        self.fouling_model = lgb.LGBMRegressor(random_state=42)
        self.features = ['feed_pressure', 'feed_flow', 'chemical_dosage', 'seawater_temp', 'feed_salinity']
        self.is_trained = False

    def train(self, data_path='desal_data.csv'):
        """Trains the three predictive models."""
        print("Training predictive models...")
        data = pd.read_csv(data_path)
        
        X = data[self.features]
        y_energy = data['energy_kwh_per_m3']
        y_quality = data['product_tds_ppm']
        y_fouling = data['fouling_rate_index_per_hr']

        # Split data for validation
        X_train, X_test, y_e_train, y_e_test = train_test_split(X, y_energy, test_size=0.2, random_state=42)
        _, _, y_q_train, y_q_test = train_test_split(X, y_quality, test_size=0.2, random_state=42)
        _, _, y_f_train, y_f_test = train_test_split(X, y_fouling, test_size=0.2, random_state=42)

        # Train models
        self.energy_model.fit(X_train, y_e_train)
        self.quality_model.fit(X_train, y_q_train)
        self.fouling_model.fit(X_train, y_f_train)
        
        # Evaluate models (for demonstration)
        print(f"Energy Model MAE: {mean_absolute_error(y_e_test, self.energy_model.predict(X_test)):.4f}")
        print(f"Quality Model MAE: {mean_absolute_error(y_q_test, self.quality_model.predict(X_test)):.4f}")
        print(f"Fouling Model MAE: {mean_absolute_error(y_f_test, self.fouling_model.predict(X_test)):.6f}")

        self.is_trained = True
        print("Models trained successfully.")
        self.save_models()

    def predict(self, input_df):
        """Predicts outputs for a given set of inputs."""
        if not self.is_trained:
            self.load_models()
            if not self.is_trained:
                raise RuntimeError("Models are not trained and no saved models were found.")

        input_df = input_df[self.features]
        energy_pred = self.energy_model.predict(input_df)[0]
        quality_pred = self.quality_model.predict(input_df)[0]
        fouling_pred = self.fouling_model.predict(input_df)[0]
        
        return energy_pred, quality_pred, fouling_pred

    def save_models(self):
        """Saves the trained models to disk."""
        joblib.dump(self.energy_model, 'energy_model.pkl')
        joblib.dump(self.quality_model, 'quality_model.pkl')
        joblib.dump(self.fouling_model, 'fouling_model.pkl')
        print("Models saved.")

    def load_models(self):
        """Loads pre-trained models from disk."""
        try:
            self.energy_model = joblib.load('energy_model.pkl')
            self.quality_model = joblib.load('quality_model.pkl')
            self.fouling_model = joblib.load('fouling_model.pkl')
            self.is_trained = True
            print("Pre-trained models loaded.")
        except FileNotFoundError:
            print("No saved models found. Please train the models first.")

if __name__ == '__main__':
    # This block allows for easy training from the command line
    predictor = DesalinationPredictor()
    predictor.train()