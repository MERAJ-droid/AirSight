"""
Test script to run the pollution source predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pollution_source_predictor import main as run_predictor

if __name__ == "__main__":
    print("Starting Pollution Source Predictor...")
    print("Make sure your CSV files are in the 'source' folder:")
    print("- source/bengaluru_nh3_emissions.csv")
    print("- source/bengaluru_pm25_emissions.csv") 
    print("- source/bengaluru_pm10_emissions.csv")
    print("- source/bengaluru_so2_emissions.csv")
    print("- source/bengaluru_combined_pollutants.csv")
    print("\n" + "="*60 + "\n")
    
    try:
        predictor = run_predictor()
        print("\nTraining completed successfully!")
        
        # Additional example predictions
        print("\n" + "="*50)
        print("ADDITIONAL EXAMPLE PREDICTIONS")
        print("="*50)
        
        # Example 1: High PM2.5 scenario
        high_pm25_scenario = [12.75, 77.45, 300, 150, 80, 50, 30, 25]  # lat, lon, and pollutant values
        
        # Example 2: High traffic scenario
        traffic_scenario = [12.85, 77.55, 200, 100, 60, 120, 80, 45]
        
        # Example 3: Industrial scenario  
        industrial_scenario = [12.90, 77.65, 400, 200, 150, 200, 150, 35]
        
        scenarios = [
            ("High PM2.5", high_pm25_scenario),
            ("Traffic Heavy", traffic_scenario), 
            ("Industrial", industrial_scenario)
        ]
        
        for name, scenario in scenarios:
            try:
                print(f"\n{name} Scenario:")
                print(f"Input values: {scenario}")
                
                # Classification
                class_result = predictor.predict_source_from_pollutants(scenario, 'classification')
                print(f"Predicted Source: {class_result['dominant_source']}")
                print(f"Confidence: {class_result['confidence']:.4f}")
                
                # Show top 3 source probabilities
                sorted_probs = sorted(class_result['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                print("Top 3 source probabilities:")
                for source, prob in sorted_probs[:3]:
                    print(f"  {source}: {prob:.4f}")
                    
            except Exception as e:
                print(f"Error with {name} scenario: {e}")
        
    except Exception as e:
        print(f"Error running predictor: {e}")
        print("\nPlease make sure:")
        print("1. All CSV files are in the correct location")
        print("2. Required Python packages are installed (pandas, sklearn, matplotlib, seaborn)")
        print("3. CSV files have the expected column structure")
