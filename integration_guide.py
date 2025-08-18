"""
Integration Guide: How to Connect Your Existing Models to the Complete System

This guide shows how to integrate your trained models from Bengaluru_code.ipynb 
and Bengaluru_2nd paper_code.ipynb into the complete air quality system.
"""

# ============================================================================
# STEP 1: Save Your Trained Models
# ============================================================================

# In your Jupyter notebooks, after training, save the models:

# For Bengaluru_code.ipynb (Single Output Model):
"""
# At the end of your notebook, add:
model.save('models/single_output_model.h5')
print("Single output model saved!")

# If you have a scaler:
import joblib
joblib.dump(scaler, 'models/single_scaler.pkl')
"""

# For Bengaluru_2nd paper_code.ipynb (Multi-Task Model):
"""
# At the end of your notebook, add:
model.save('models/multi_task_model.h5')
print("Multi-task model saved!")

# If you have a scaler:
import joblib
joblib.dump(scaler, 'models/multi_scaler.pkl')
"""

# ============================================================================
# STEP 2: Model Loading Integration
# ============================================================================

# Replace the load_models() method in complete_air_quality_system.py:

def load_models_integrated(self):
    """Load all required models - INTEGRATED VERSION"""
    try:
        print("🔄 Loading models...")
        
        # Load Single Output Model
        try:
            import tensorflow as tf
            self.single_model = tf.keras.models.load_model('models/single_output_model.h5')
            print("✅ Single output model loaded")
        except Exception as e:
            print(f"⚠️ Single model not found: {e}")
            self.single_model = None
        
        # Load Multi-Task Model
        try:
            self.multi_model = tf.keras.models.load_model('models/multi_task_model.h5')
            print("✅ Multi-task model loaded")
        except Exception as e:
            print(f"⚠️ Multi-task model not found: {e}")
            self.multi_model = None
        
        # Load Scalers (if you use them)
        try:
            import joblib
            self.single_scaler = joblib.load('models/single_scaler.pkl')
            self.multi_scaler = joblib.load('models/multi_scaler.pkl')
            print("✅ Scalers loaded")
        except Exception as e:
            print(f"⚠️ Scalers not found: {e}")
            self.single_scaler = None
            self.multi_scaler = None
        
        # Load pollution source predictor
        print("Loading pollution source predictor...")
        datasets = load_emission_datasets()
        if datasets:
            self.source_predictor = PollutionSourcePredictor()
            if 'pm25' in datasets:
                main_data = datasets['pm25']
            else:
                main_data = list(datasets.values())[0]
            
            enhanced_data = self.source_predictor.create_synthetic_pollutant_data(main_data)
            X, y = self.source_predictor.prepare_training_data(enhanced_data)
            self.source_predictor.train_classification_model(X, y)
            self.source_predictor.train_regression_model(X, y)
            print("✅ Pollution source predictor loaded successfully")
        
        self.update_status("✅ Models loaded successfully")
        
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        self.update_status(f"❌ Error loading models: {e}")

# ============================================================================
# STEP 3: Real Prediction Methods
# ============================================================================

# Replace the prediction methods with real implementations:

def predict_pollutants_single_real(self, image_path):
    """Real prediction using your single output model"""
    try:
        if self.single_model is None:
            raise Exception("Single output model not loaded")
        
        # Preprocess image for your model
        image_array = self.preprocess_image(image_path, (224, 224))
        
        # Make prediction
        prediction = self.single_model.predict(image_array)[0]
        
        # Map predictions to pollutant names
        # Adjust these based on your model's output order
        pollutants = {
            'PM2.5': float(prediction[0]),
            'PM10': float(prediction[1]),
            'NO2': float(prediction[2]),
            'SO2': float(prediction[3]),
            'CO': float(prediction[4]),
            'O3': float(prediction[5]),
            'AQI': float(prediction[6]) if len(prediction) > 6 else 0
        }
        
        # Calculate AQI if not predicted directly
        if pollutants['AQI'] == 0:
            pollutants['AQI'] = max(
                pollutants['PM2.5'] * 2, 
                pollutants['PM10'] * 1.5, 
                pollutants['NO2'] * 1.8
            )
        
        return pollutants
        
    except Exception as e:
        raise Exception(f"Single model prediction failed: {e}")

def predict_pollutants_multi_real(self, image_path):
    """Real prediction using your multi-task model"""
    try:
        if self.multi_model is None:
            raise Exception("Multi-task model not loaded")
        
        # Preprocess image for your model (224x112 for multi-task)
        image_array = self.preprocess_image(image_path, (112, 224))
        
        # Make prediction
        predictions = self.multi_model.predict(image_array)
        
        # Extract predictions from multiple outputs
        # Adjust based on your model's output structure
        if isinstance(predictions, list):
            # Multiple outputs
            output1 = predictions[0][0]  # First output (e.g., 2 parameters)
            output2 = predictions[1][0]  # Second output (e.g., 5 parameters)
            
            pollutants = {
                'PM2.5': float(output1[0]),
                'PM10': float(output1[1]),
                'NO2': float(output2[0]),
                'SO2': float(output2[1]),
                'CO': float(output2[2]),
                'O3': float(output2[3]),
                'AQI': float(output2[4]) if len(output2) > 4 else 0
            }
        else:
            # Single output with multiple values
            prediction = predictions[0]
            pollutants = {
                'PM2.5': float(prediction[0]),
                'PM10': float(prediction[1]),
                'NO2': float(prediction[2]),
                'SO2': float(prediction[3]),
                'CO': float(prediction[4]),
                'O3': float(prediction[5]),
                'AQI': float(prediction[6]) if len(prediction) > 6 else 0
            }
        
        # Calculate AQI if not predicted directly
        if pollutants['AQI'] == 0:
            pollutants['AQI'] = max(
                pollutants['PM2.5'] * 2, 
                pollutants['PM10'] * 1.5, 
                pollutants['NO2'] * 1.8
            )
        
        return pollutants
        
    except Exception as e:
        raise Exception(f"Multi-task model prediction failed: {e}")

# ============================================================================
# STEP 4: File Structure Setup
# ============================================================================

"""
Create this file structure in your project:

Bengaluru/
├── complete_air_quality_system.py
├── fixed_source_predictor.py
├── launch_airsight.py
├── models/                          # Create this folder
│   ├── single_output_model.h5       # Save from Bengaluru_code.ipynb
│   ├── multi_task_model.h5          # Save from Bengaluru_2nd paper_code.ipynb
│   ├── single_scaler.pkl            # If you use scalers
│   └── multi_scaler.pkl
├── source/                          # Your existing folder
│   ├── bengaluru_nh3_emissions.csv
│   ├── bengaluru_pm25_emissions.csv
│   ├── bengaluru_pm10_emissions.csv
│   └── bengaluru_so2_emissions.csv
├── test_images/                     # Create for test images
│   └── (your test images)
└── ... (other files)
"""

# ============================================================================
# STEP 5: Integration Steps
# ============================================================================

"""
1. Save your trained models from Jupyter notebooks
2. Create the models/ folder
3. Replace the methods in complete_air_quality_system.py
4. Test with: python launch_airsight.py
5. Load test images and verify predictions

Example integration in your Jupyter notebook:

# At the end of training in Bengaluru_code.ipynb:
model.save('../models/single_output_model.h5')
print("Model saved for integration!")

# Test the integration:
import sys
sys.path.append('..')
from complete_air_quality_system import CompleteAirQualitySystem

# Create instance and test
app = CompleteAirQualitySystem()
result = app.predict_pollutants_single_real('test_image.jpg')
print("Integration test:", result)
"""

# ============================================================================
# STEP 6: Debugging Tips
# ============================================================================

"""
Common Issues and Solutions:

1. Model Loading Errors:
   - Check file paths are correct
   - Ensure models were saved properly
   - Verify TensorFlow version compatibility

2. Shape Mismatch Errors:
   - Check image preprocessing dimensions
   - Verify input shape matches training
   - Ensure PIL resize format (width, height)

3. Prediction Format Issues:
   - Check model output structure
   - Adjust pollutant mapping
   - Handle multiple outputs correctly

4. Performance Issues:
   - Use model.predict() for single images
   - Consider batch processing for multiple images
   - Cache model loading for better performance

Debugging Code:
"""

def debug_model_integration():
    """Debug function to test model integration"""
    print("=== Model Integration Debug ===")
    
    # Test 1: Check file existence
    import os
    files_to_check = [
        'models/single_output_model.h5',
        'models/multi_task_model.h5',
        'fixed_source_predictor.py'
    ]
    
    for file_path in files_to_check:
        if os.path.exists(file_path):
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
    
    # Test 2: Try loading models
    try:
        import tensorflow as tf
        model = tf.keras.models.load_model('models/single_output_model.h5')
        print(f"✅ Single model loaded: {model.input_shape}")
    except Exception as e:
        print(f"❌ Single model error: {e}")
    
    # Test 3: Test image preprocessing
    try:
        from PIL import Image
        import numpy as np
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(dummy_image)
        img_resized = img.resize((224, 224))
        print(f"✅ Image preprocessing works: {np.array(img_resized).shape}")
    except Exception as e:
        print(f"❌ Image preprocessing error: {e}")

if __name__ == "__main__":
    debug_model_integration()
