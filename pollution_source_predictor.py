import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class PollutionSourcePredictor:
    def __init__(self):
        self.classifier = None
        self.regressor = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_columns = []
        self.source_columns = []
        
    def load_and_combine_data(self):
        """Load all emission datasets and combine them"""
        try:
            # Load datasets
            nh3_df = pd.read_csv('source/bengaluru_nh3_emissions.csv')
            pm25_df = pd.read_csv('source/bengaluru_pm25_emissions.csv')
            pm10_df = pd.read_csv('source/bengaluru_pm10_emissions.csv')
            so2_df = pd.read_csv('source/bengaluru_so2_emissions.csv')
            combined_df = pd.read_csv('source/bengaluru_combined_pollutants.csv')
            
            print("Successfully loaded all datasets")
            print(f"NH3 dataset shape: {nh3_df.shape}")
            print(f"PM2.5 dataset shape: {pm25_df.shape}")
            print(f"PM10 dataset shape: {pm10_df.shape}")
            print(f"SO2 dataset shape: {so2_df.shape}")
            print(f"Combined dataset shape: {combined_df.shape}")
            
            # Use PM2.5 dataset as base (it seems to have the most complete source information)
            return pm25_df
            
        except Exception as e:
            print(f"Error loading datasets: {e}")
            return None
    
    def prepare_data(self, df, prediction_type='classification'):
        """Prepare features and labels for training"""
        # Identify pollutant columns (features)
        pollutant_cols = [col for col in df.columns if 'total' in col.lower() or 
                         col in ['lat', 'lon']]
        
        # Identify source columns (targets)
        source_cols = [col for col in df.columns if 'HTAPv3' in col]
        
        print(f"Pollutant columns (features): {pollutant_cols}")
        print(f"Source columns: {source_cols}")
        
        # Remove rows with missing values
        df_clean = df.dropna()
        print(f"Dataset shape after removing NaN: {df_clean.shape}")
        
        # Features (pollutant values)
        X = df_clean[pollutant_cols]
        
        if prediction_type == 'classification':
            # For classification: find dominant source
            source_data = df_clean[source_cols]
            # Get the column name with maximum value for each row
            y = source_data.idxmax(axis=1)
            # Encode source names to numbers
            y_encoded = self.label_encoder.fit_transform(y)
            self.feature_columns = pollutant_cols
            self.source_columns = source_cols
            return X, y_encoded, y
            
        elif prediction_type == 'regression':
            # For regression: predict all source values
            y = df_clean[source_cols]
            self.feature_columns = pollutant_cols
            self.source_columns = source_cols
            return X, y, None
    
    def train_classification_model(self, X, y):
        """Train classification model to predict dominant pollution source"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        self.classifier.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.classifier.predict(X_test_scaled)
        
        # Evaluation
        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nClassification Results:")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Feature importance
        feature_importance = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.classifier.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nFeature Importance:")
        print(feature_importance)
        
        return X_test_scaled, y_test, y_pred, feature_importance
    
    def train_regression_model(self, X, y):
        """Train regression model to predict source contributions"""
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train regressor
        self.regressor = RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            max_depth=10
        )
        self.regressor.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = self.regressor.predict(X_test_scaled)
        
        # Evaluation
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"\nRegression Results:")
        print(f"Mean Squared Error: {mse:.4f}")
        print(f"R² Score: {r2:.4f}")
        
        return X_test_scaled, y_test, y_pred
    
    def predict_source_from_pollutants(self, pollutant_values, model_type='classification'):
        """Predict pollution source from pollutant values"""
        # Convert to DataFrame if it's a list/array
        if isinstance(pollutant_values, (list, np.ndarray)):
            if len(pollutant_values) != len(self.feature_columns):
                raise ValueError(f"Expected {len(self.feature_columns)} features, got {len(pollutant_values)}")
            pollutant_values = pd.DataFrame([pollutant_values], columns=self.feature_columns)
        
        # Scale the input
        pollutant_scaled = self.scaler.transform(pollutant_values)
        
        if model_type == 'classification' and self.classifier:
            # Predict dominant source
            prediction = self.classifier.predict(pollutant_scaled)[0]
            source_name = self.label_encoder.inverse_transform([prediction])[0]
            probability = self.classifier.predict_proba(pollutant_scaled)[0]
            
            return {
                'dominant_source': source_name,
                'confidence': max(probability),
                'all_probabilities': dict(zip(self.label_encoder.classes_, probability))
            }
            
        elif model_type == 'regression' and self.regressor:
            # Predict all source contributions
            predictions = self.regressor.predict(pollutant_scaled)[0]
            return dict(zip(self.source_columns, predictions))
        
        else:
            raise ValueError("Model not trained or invalid model type")
    
    def visualize_results(self, feature_importance=None):
        """Create visualizations for the results"""
        plt.figure(figsize=(15, 10))
        
        if feature_importance is not None:
            # Feature importance plot
            plt.subplot(2, 2, 1)
            sns.barplot(data=feature_importance.head(10), x='importance', y='feature')
            plt.title('Top 10 Feature Importance (Classification)')
            plt.xlabel('Importance')
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run the pollution source prediction"""
    print("=== Pollution Source Prediction from Pollutant Values ===\n")
    
    # Initialize predictor
    predictor = PollutionSourcePredictor()
    
    # Load and combine data
    df = predictor.load_and_combine_data()
    if df is None:
        print("Failed to load data. Please check file paths.")
        return
    
    print(f"\nDataset columns: {list(df.columns)}")
    print(f"Dataset shape: {df.shape}")
    
    # Train Classification Model
    print("\n" + "="*50)
    print("TRAINING CLASSIFICATION MODEL")
    print("="*50)
    
    X_class, y_class, y_class_names = predictor.prepare_data(df, 'classification')
    X_test_class, y_test_class, y_pred_class, feature_imp = predictor.train_classification_model(X_class, y_class)
    
    # Train Regression Model
    print("\n" + "="*50)
    print("TRAINING REGRESSION MODEL")
    print("="*50)
    
    X_reg, y_reg, _ = predictor.prepare_data(df, 'regression')
    X_test_reg, y_test_reg, y_pred_reg = predictor.train_regression_model(X_reg, y_reg)
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Use mean values as example
    example_pollutants = X_class.mean().values
    print(f"Example pollutant values: {example_pollutants}")
    
    # Classification prediction
    class_result = predictor.predict_source_from_pollutants(example_pollutants, 'classification')
    print(f"\nClassification Result:")
    print(f"Dominant Source: {class_result['dominant_source']}")
    print(f"Confidence: {class_result['confidence']:.4f}")
    
    # Regression prediction
    reg_result = predictor.predict_source_from_pollutants(example_pollutants, 'regression')
    print(f"\nRegression Result (Source Contributions):")
    for source, contribution in reg_result.items():
        print(f"{source}: {contribution:.4f}")
    
    # Visualize results
    predictor.visualize_results(feature_imp)
    
    print("\n=== Model Training Complete ===")
    return predictor

if __name__ == "__main__":
    predictor = main()
