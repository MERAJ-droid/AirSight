import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from pollution_source_predictor import PollutionSourcePredictor
import pickle

class CombinedPredictor:
    """
    Combined model that predicts pollution sources from images
    by chaining image-to-pollutants and pollutants-to-source models
    """
    
    def __init__(self, image_model_path=None, scaler_path=None):
        self.image_model = None
        self.source_predictor = None
        self.scaler = None
        
        # Load pre-trained models if paths provided
        if image_model_path:
            self.load_image_model(image_model_path)
        if scaler_path:
            self.load_scaler(scaler_path)
    
    def load_image_model(self, model_path):
        """Load the pre-trained image-to-pollutants model"""
        try:
            self.image_model = tf.keras.models.load_model(model_path)
            print(f"Loaded image model from {model_path}")
        except Exception as e:
            print(f"Error loading image model: {e}")
    
    def load_scaler(self, scaler_path):
        """Load the scaler used for pollutant normalization"""
        try:
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
            print(f"Loaded scaler from {scaler_path}")
        except Exception as e:
            print(f"Error loading scaler: {e}")
    
    def train_source_predictor(self):
        """Train the pollution source predictor"""
        self.source_predictor = PollutionSourcePredictor()
        
        # Load and prepare data
        df = self.source_predictor.load_and_combine_data()
        if df is None:
            raise ValueError("Failed to load emission data")
        
        # Train classification model
        X, y, _ = self.source_predictor.prepare_data(df, 'classification')
        self.source_predictor.train_classification_model(X, y)
        
        print("Source predictor trained successfully")
    
    def preprocess_image(self, image_path, target_size=(224, 112)):
        """Preprocess image for model input"""
        try:
            image = Image.open(image_path)
            image = image.resize(target_size)  # Resize to match your model's expected input
            image_array = np.array(image) / 255.0  # Normalize
            image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
            return image_array
        except Exception as e:
            print(f"Error preprocessing image: {e}")
            return None
    
    def predict_from_image(self, image_path, return_intermediate=False):
        """
        Predict pollution source from image through the complete pipeline
        
        Args:
            image_path: Path to the input image
            return_intermediate: If True, also return intermediate pollutant predictions
        
        Returns:
            Dictionary with source predictions and optionally pollutant values
        """
        if self.image_model is None:
            raise ValueError("Image model not loaded. Please load or train the model first.")
        
        if self.source_predictor is None:
            raise ValueError("Source predictor not trained. Please train it first.")
        
        # Step 1: Preprocess image
        image_array = self.preprocess_image(image_path)
        if image_array is None:
            return None
        
        # Step 2: Predict pollutants from image
        if hasattr(self.image_model, 'predict'):
            # Check if it's a multi-output model
            try:
                predictions = self.image_model.predict(image_array)
                if isinstance(predictions, list):
                    # Multi-output model (like your 2nd paper model)
                    y_pred_2, y_pred_5 = predictions
                    # Reconstruct full pollutant array
                    first_column = y_pred_5[:, 0:1]
                    y_pred_first_three = np.concatenate((first_column, y_pred_2), axis=1)
                    remaining_columns = y_pred_5[:, 1:]
                    pollutants = np.concatenate((y_pred_first_three, remaining_columns), axis=1)
                else:
                    # Single output model
                    pollutants = predictions
            except Exception as e:
                print(f"Error predicting from image: {e}")
                return None
        else:
            raise ValueError("Invalid image model")
        
        # Step 3: Inverse transform if scaler is available
        if self.scaler is not None:
            pollutants = self.scaler.inverse_transform(pollutants)
        
        # Step 4: Predict source from pollutants
        # Note: You may need to map your pollutant order to the source predictor's expected format
        try:
            # Assuming pollutants are in order: [AQI, PM2.5, PM10, O3, CO, SO2, NO2]
            # Map to source predictor's expected features (you may need to adjust this)
            source_prediction = self.source_predictor.predict_source_from_pollutants(
                pollutants[0], 'classification'
            )
        except Exception as e:
            print(f"Error predicting source: {e}")
            return None
        
        result = {
            'dominant_source': source_prediction['dominant_source'],
            'confidence': source_prediction['confidence'],
            'all_source_probabilities': source_prediction['all_probabilities']
        }
        
        if return_intermediate:
            result['pollutant_values'] = {
                'AQI': float(pollutants[0][0]),
                'PM2.5': float(pollutants[0][1]),
                'PM10': float(pollutants[0][2]),
                'O3': float(pollutants[0][3]),
                'CO': float(pollutants[0][4]),
                'SO2': float(pollutants[0][5]),
                'NO2': float(pollutants[0][6])
            }
        
        return result
    
    def predict_from_pollutants(self, pollutant_dict):
        """
        Predict pollution source directly from pollutant values
        
        Args:
            pollutant_dict: Dictionary with pollutant values
                           e.g., {'AQI': 150, 'PM2.5': 60, 'PM10': 80, ...}
        """
        if self.source_predictor is None:
            raise ValueError("Source predictor not trained.")
        
        # Convert dict to array in expected order
        pollutant_values = [
            pollutant_dict.get('AQI', 0),
            pollutant_dict.get('PM2.5', 0),
            pollutant_dict.get('PM10', 0),
            pollutant_dict.get('O3', 0),
            pollutant_dict.get('CO', 0),
            pollutant_dict.get('SO2', 0),
            pollutant_dict.get('NO2', 0)
        ]
        
        return self.source_predictor.predict_source_from_pollutants(
            pollutant_values, 'classification'
        )

def main():
    """Example usage of the combined predictor"""
    print("=== Combined Image-to-Source Pollution Predictor ===\n")
    
    # Initialize combined predictor
    predictor = CombinedPredictor()
    
    # Train the source predictor
    print("Training source predictor...")
    predictor.train_source_predictor()
    
    # Example: Predict from pollutant values directly
    print("\n--- Example: Predict from pollutant values ---")
    example_pollutants = {
        'AQI': 150,
        'PM2.5': 60,
        'PM10': 80,
        'O3': 100,
        'CO': 2.5,
        'SO2': 15,
        'NO2': 40
    }
    
    result = predictor.predict_from_pollutants(example_pollutants)
    print(f"Input pollutants: {example_pollutants}")
    print(f"Predicted source: {result['dominant_source']}")
    print(f"Confidence: {result['confidence']:.4f}")
    
    # If you have a trained image model, you can also predict from images:
    # result = predictor.predict_from_image('path/to/image.jpg', return_intermediate=True)
    
    print("\n=== Setup Complete ===")
    print("To use with images, load your trained image model using:")
    print("predictor.load_image_model('path/to/your/model.h5')")
    print("predictor.load_scaler('path/to/your/scaler.pkl')")

if __name__ == "__main__":
    main()
