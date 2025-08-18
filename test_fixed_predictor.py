"""
Test script for the fixed pollution source predictor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from fixed_source_predictor import main as run_predictor

if __name__ == "__main__":
    print("Starting Fixed Pollution Source Predictor...")
    print("This version uses realistic pollutant measurements as features!")
    print("\n" + "="*60 + "\n")
    
    try:
        predictor = run_predictor()
        
        if predictor is None:
            print("Failed to train predictor")
            exit(1)
            
        print("\nTraining completed successfully!")
        
        # Test with realistic pollution scenarios
        print("\n" + "="*50)
        print("REALISTIC POLLUTION SCENARIOS")
        print("="*50)
        
        # Format: [AQI, PM2.5, PM10, NO2, SO2, CO, O3]
        scenarios = [
            {
                'name': 'Traffic Heavy Area',
                'values': [120, 65, 85, 80, 15, 12, 35],
                'description': 'High NO2 and CO from vehicles'
            },
            {
                'name': 'Industrial Zone', 
                'values': [180, 95, 130, 70, 45, 8, 25],
                'description': 'High PM and SO2 from industry'
            },
            {
                'name': 'Residential Cooking',
                'values': [90, 45, 60, 25, 8, 15, 40],
                'description': 'Moderate PM, high CO from cooking'
            },
            {
                'name': 'Agricultural Burning',
                'values': [200, 150, 180, 40, 20, 25, 30],
                'description': 'Very high PM from biomass burning'
            },
            {
                'name': 'Clean Air Day',
                'values': [35, 15, 25, 20, 5, 3, 60],
                'description': 'Low pollution, good air quality'
            }
        ]
        
        for scenario in scenarios:
            print(f"\n{'='*20} {scenario['name']} {'='*20}")
            print(f"Description: {scenario['description']}")
            print(f"Pollutant values: {scenario['values']}")
            print("Format: [AQI, PM2.5, PM10, NO2, SO2, CO, O3]")
            
            try:
                result = predictor.predict_source_from_pollutants(scenario['values'])
                
                if 'classification' in result:
                    source_name = result['classification']['dominant_source']
                    confidence = result['classification']['confidence']
                    
                    # Clean up source name for display
                    clean_name = source_name.replace('HTAPv3_', '').replace('_', ' ').title()
                    print(f"\n🎯 Predicted Source: {clean_name}")
                    print(f"📊 Confidence: {confidence:.1%}")
                    
                    # Show top 3 probabilities
                    print("\n📈 Top 3 source probabilities:")
                    sorted_probs = sorted(result['classification']['all_probabilities'].items(), 
                                        key=lambda x: x[1], reverse=True)
                    for i, (source, prob) in enumerate(sorted_probs[:3], 1):
                        clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                        print(f"  {i}. {clean_source}: {prob:.1%}")
                
                if 'regression' in result:
                    print(f"\n🏭 Top 3 emission contributions:")
                    sorted_emissions = sorted(result['regression'].items(), 
                                            key=lambda x: x[1], reverse=True)
                    for i, (source, emission) in enumerate(sorted_emissions[:3], 1):
                        if emission > 0.1:  # Only show significant contributions
                            clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                            print(f"  {i}. {clean_source}: {emission:.1f} units")
                            
            except Exception as e:
                print(f"❌ Error predicting for {scenario['name']}: {e}")
        
        print(f"\n{'='*60}")
        print("🎉 All predictions completed!")
        print("💡 The model now uses realistic pollutant measurements")
        print("🔬 Features: AQI, PM2.5, PM10, NO2, SO2, CO, O3")
        print("🏭 Predicts emission sources from these measurements")
        
    except Exception as e:
        print(f"❌ Error running predictor: {e}")
        import traceback
        traceback.print_exc()
