"""
Fixed Pollution Source Predictor that uses realistic pollutant measurements
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

class PollutionSourcePredictor:
    def __init__(self):
        self.classification_model = None
        self.regression_model = None
        self.scaler = StandardScaler()
        self.source_columns = []
        self.pollutant_columns = []
        
    def create_synthetic_pollutant_data(self, emission_data):
        """
        Create realistic pollutant concentration data from emission data
        This simulates what sensors would actually measure
        """
        print("Creating synthetic pollutant measurements from emission data...")
        print(f"Input data shape: {emission_data.shape}")
        print(f"Available columns: {list(emission_data.columns)}")
        
        # Create realistic pollutant concentrations based on emissions
        # These would be the values your image model would predict
        synthetic_data = emission_data.copy()
        
        # Get emission columns with safe defaults
        traffic_emissions = emission_data.get('HTAPv3_5_1_Road_Transport', pd.Series(0, index=emission_data.index))
        industry_emissions = emission_data.get('HTAPv3_4_1_Industry', pd.Series(0, index=emission_data.index))
        energy_emissions = emission_data.get('HTAPv3_3_Energy', pd.Series(0, index=emission_data.index))
        residential_emissions = emission_data.get('HTAPv3_6_Residential', pd.Series(0, index=emission_data.index))
        
        # Convert emissions to concentration estimates (simplified model)
        # PM2.5 concentration (µg/m³) - based on total emissions and local factors
        if 'total_pm25_emissions' in emission_data.columns:
            synthetic_data['PM25_concentration'] = (
                emission_data['total_pm25_emissions'] * 0.1 + 
                np.random.normal(0, 5, len(emission_data))
            ).clip(0, None)
        else:
            # Create from other sources if total not available
            synthetic_data['PM25_concentration'] = (
                (traffic_emissions + industry_emissions) * 0.05 + 
                np.random.normal(10, 5, len(emission_data))
            ).clip(0, None)
        
        # PM10 concentration (µg/m³)
        if 'total_pm10_emissions' in emission_data.columns:
            synthetic_data['PM10_concentration'] = (
                emission_data['total_pm10_emissions'] * 0.12 + 
                np.random.normal(0, 8, len(emission_data))
            ).clip(0, None)
        else:
            synthetic_data['PM10_concentration'] = (
                synthetic_data['PM25_concentration'] * 1.5 + 
                np.random.normal(5, 8, len(emission_data))
            ).clip(0, None)
        
        # NO2 concentration (µg/m³) - mainly from traffic and industry
        synthetic_data['NO2_concentration'] = (
            traffic_emissions * 0.15 + industry_emissions * 0.08 + 
            np.random.normal(15, 3, len(emission_data))
        ).clip(0, None)
        
        # SO2 concentration (µg/m³) - mainly from industry and energy
        synthetic_data['SO2_concentration'] = (
            industry_emissions * 0.05 + energy_emissions * 0.12 + 
            np.random.normal(5, 2, len(emission_data))
        ).clip(0, None)
        
        # CO concentration (mg/m³) - mainly from traffic and residential
        synthetic_data['CO_concentration'] = (
            traffic_emissions * 0.02 + residential_emissions * 0.03 + 
            np.random.normal(2, 0.5, len(emission_data))
        ).clip(0, None)
        
        # O3 concentration (µg/m³) - formed from precursors
        synthetic_data['O3_concentration'] = (
            (traffic_emissions + industry_emissions) * 0.08 + 
            np.random.normal(50, 10, len(emission_data))
        ).clip(0, None)
        
        # AQI calculation (simplified)
        pm25_aqi = synthetic_data['PM25_concentration'] * 2
        pm10_aqi = synthetic_data['PM10_concentration'] * 1.5
        no2_aqi = synthetic_data['NO2_concentration'] * 1.8
        
        # Calculate AQI as maximum of component AQIs
        synthetic_data['AQI'] = np.maximum(np.maximum(pm25_aqi, pm10_aqi), no2_aqi)
        
        print(f"Created synthetic data shape: {synthetic_data.shape}")
        print(f"Sample pollutant concentrations:")
        pollutant_cols = ['AQI', 'PM25_concentration', 'PM10_concentration', 'NO2_concentration', 'SO2_concentration', 'CO_concentration', 'O3_concentration']
        available_pollutants = [col for col in pollutant_cols if col in synthetic_data.columns]
        if available_pollutants:
            print(synthetic_data[available_pollutants].head())
        
        return synthetic_data
    
    def prepare_training_data(self, data):
        """
        Prepare features (pollutant concentrations) and targets (emission sources)
        """
        # Features: What sensors/images would actually measure
        pollutant_features = [
            'AQI', 'PM25_concentration', 'PM10_concentration', 
            'NO2_concentration', 'SO2_concentration', 'CO_concentration', 'O3_concentration'
        ]
        
        # Filter features that exist in the data
        available_features = [col for col in pollutant_features if col in data.columns]
        
        if not available_features:
            raise ValueError("No pollutant concentration features found in data")
        
        # Source columns (what we want to predict)
        source_columns = [col for col in data.columns if col.startswith('HTAPv3_')]
        
        if not source_columns:
            raise ValueError("No emission source columns found in data")
        
        self.pollutant_columns = available_features
        self.source_columns = source_columns
        
        print(f"Using pollutant features: {available_features}")
        print(f"Predicting emission sources: {len(source_columns)} sources")
        
        return data[available_features], data[source_columns]
    
    def train_classification_model(self, X, y):
        """
        Train model to predict dominant pollution source
        """
        print("\n" + "="*50)
        print("TRAINING CLASSIFICATION MODEL")
        print("="*50)
        
        # Create dominant source labels
        y_dominant = y.idxmax(axis=1)
        
        # Remove rows where all sources are 0
        non_zero_mask = y.sum(axis=1) > 0
        X_clean = X[non_zero_mask]
        y_clean = y_dominant[non_zero_mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid training data after removing zero emissions")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42, stratify=y_clean
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.classification_model = RandomForestClassifier(
            n_estimators=100, random_state=42, max_depth=10
        )
        self.classification_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.classification_model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"Classification Accuracy: {accuracy:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.pollutant_columns,
            'importance': self.classification_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        # Plot feature importance
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance, x='importance', y='feature')
        plt.title('Pollutant Feature Importance for Source Classification')
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
        
        return accuracy
    
    def train_regression_model(self, X, y):
        """
        Train model to predict emission amounts for each source
        """
        print("\n" + "="*50)
        print("TRAINING REGRESSION MODEL")
        print("="*50)
        
        # Remove rows where all sources are 0
        non_zero_mask = y.sum(axis=1) > 0
        X_clean = X[non_zero_mask]
        y_clean = y[non_zero_mask]
        
        if len(X_clean) == 0:
            raise ValueError("No valid training data after removing zero emissions")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_clean, y_clean, test_size=0.3, random_state=42
        )
        
        # Scale features (use same scaler as classification)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.regression_model = RandomForestRegressor(
            n_estimators=100, random_state=42, max_depth=15
        )
        self.regression_model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.regression_model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Regression MSE: {mse:.4f}")
        print(f"Regression R² Score: {r2:.4f}")
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        return r2
    
    def predict_source_from_pollutants(self, pollutant_values, prediction_type='both'):
        """
        Predict pollution source from pollutant measurements
        
        Args:
            pollutant_values: List or array of pollutant concentrations
                             [AQI, PM2.5, PM10, NO2, SO2, CO, O3]
            prediction_type: 'classification', 'regression', or 'both'
        """
        if self.classification_model is None and self.regression_model is None:
            raise ValueError("Models not trained yet")
        
        # Convert to DataFrame for easier handling
        if isinstance(pollutant_values, (list, tuple)):
            pollutant_values = np.array(pollutant_values).reshape(1, -1)
        
        # Ensure we have the right number of features
        if pollutant_values.shape[1] != len(self.pollutant_columns):
            raise ValueError(f"Expected {len(self.pollutant_columns)} features, got {pollutant_values.shape[1]}")
        
        # Scale the input
        pollutant_values_scaled = self.scaler.transform(pollutant_values)
        
        results = {}
        
        if prediction_type in ['classification', 'both'] and self.classification_model is not None:
            # Classification prediction
            class_pred = self.classification_model.predict(pollutant_values_scaled)[0]
            class_proba = self.classification_model.predict_proba(pollutant_values_scaled)[0]
            
            # Get class names
            class_names = self.classification_model.classes_
            proba_dict = dict(zip(class_names, class_proba))
            
            results['classification'] = {
                'dominant_source': class_pred,
                'confidence': max(class_proba),
                'all_probabilities': proba_dict
            }
        
        if prediction_type in ['regression', 'both'] and self.regression_model is not None:
            # Regression prediction
            reg_pred = self.regression_model.predict(pollutant_values_scaled)[0]
            
            results['regression'] = dict(zip(self.source_columns, reg_pred))
        
        return results

def load_emission_datasets():
    """Load all emission datasets"""
    datasets = {}
    source_folder = 'source'
    
    file_mappings = {
        'nh3': 'bengaluru_nh3_emissions.csv',
        'pm25': 'bengaluru_pm25_emissions.csv',
        'pm10': 'bengaluru_pm10_emissions.csv',
        'so2': 'bengaluru_so2_emissions.csv'
    }
    
    for pollutant, filename in file_mappings.items():
        filepath = os.path.join(source_folder, filename)
        if os.path.exists(filepath):
            try:
                df = pd.read_csv(filepath)
                datasets[pollutant] = df
                print(f"✓ Loaded {pollutant}: {df.shape}")
            except Exception as e:
                print(f"✗ Error loading {pollutant}: {e}")
        else:
            print(f"✗ File not found: {filepath}")
    
    return datasets

def main():
    """Main function to run the pollution source predictor"""
    print("=== Pollution Source Prediction from Pollutant Values ===")
    print()
    
    try:
        # Load datasets
        datasets = load_emission_datasets()
        
        if not datasets:
            raise ValueError("No datasets loaded successfully")
        
        # Use PM2.5 dataset as the main dataset (it has the most complete data)
        if 'pm25' in datasets:
            main_data = datasets['pm25']
            print(f"Using PM2.5 dataset as main dataset: {main_data.shape}")
        else:
            # Use any available dataset
            main_data = list(datasets.values())[0]
            print(f"Using available dataset: {main_data.shape}")
        
        # Initialize predictor
        predictor = PollutionSourcePredictor()
        
        # Create synthetic pollutant measurements
        enhanced_data = predictor.create_synthetic_pollutant_data(main_data)
        
        # Prepare training data
        X, y = predictor.prepare_training_data(enhanced_data)
        
        print(f"\nTraining data shape: X={X.shape}, y={y.shape}")
        print(f"Sample pollutant values:\n{X.head()}")
        
        # Train models
        class_score = predictor.train_classification_model(X, y)
        reg_score = predictor.train_regression_model(X, y)
        
        # Example predictions
        print("\n" + "="*50)
        print("EXAMPLE PREDICTIONS")
        print("="*50)
        
        # Example 1: High pollution scenario
        high_pollution = [150, 80, 120, 60, 25, 8, 45]  # AQI, PM2.5, PM10, NO2, SO2, CO, O3
        print(f"\nHigh Pollution Scenario: {high_pollution}")
        
        result = predictor.predict_source_from_pollutants(high_pollution)
        
        if 'classification' in result:
            print(f"Dominant Source: {result['classification']['dominant_source']}")
            print(f"Confidence: {result['classification']['confidence']:.4f}")
            
            print("\nTop 3 source probabilities:")
            sorted_probs = sorted(result['classification']['all_probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)
            for source, prob in sorted_probs[:3]:
                print(f"  {source.replace('HTAPv3_', '').replace('_', ' ')}: {prob:.4f}")
        
        if 'regression' in result:
            print(f"\nTop 3 emission contributions:")
            sorted_emissions = sorted(result['regression'].items(), 
                                    key=lambda x: x[1], reverse=True)
            for source, emission in sorted_emissions[:3]:
                print(f"  {source.replace('HTAPv3_', '').replace('_', ' ')}: {emission:.2f}")
        
        return predictor
        
    except Exception as e:
        print(f"Error in main execution: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    predictor = main()
