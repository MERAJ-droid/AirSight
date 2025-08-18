"""
Launcher for the Complete Air Quality Analysis System
"""

import sys
import os

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from enhanced_air_quality_system import main
    
    print("🚀 Launching Enhanced AirSight - Air Quality Analysis System")
    print("=" * 70)
    print()
    print("🎨 ENHANCED FEATURES:")
    print("───────────────────")
    print("📸 Advanced Image Analysis - Professional image loading & display")
    print("🌫️ Pollutant Detection - ML-powered prediction of 7 key pollutants")
    print("🏭 Source Identification - AI identifies pollution emission sources")
    print("🏥 Health Assessment - Comprehensive health impact & recommendations")
    print("📊 Interactive Visualizations - Professional charts & graphs")
    print("🎨 Enhanced UI - Dark theme, better readability, modern design")
    print()
    print("🤖 AI MODELS:")
    print("─────────────")
    print("🔬 Single Output Model - Unified pollutant prediction")
    print("🔬 Multi-Task Model - Specialized parameter branches")
    print("🔬 Source Predictor - Advanced emission source mapping")
    print()
    print("💡 IMPROVEMENTS:")
    print("───────────────")
    print("✅ Dark theme for better visibility")
    print("✅ Enhanced text formatting and colors")
    print("✅ Larger fonts and better spacing")
    print("✅ Professional status updates")
    print("✅ Improved visualizations")
    print("✅ Better error handling")
    print()
    print("Loading enhanced application...")
    print("=" * 70)
    
    # Run the enhanced application
    main()
    
except ImportError as e:
    print(f"❌ Import Error: {e}")
    print("\nMissing dependencies. Please install:")
    print("pip install tensorflow pandas scikit-learn matplotlib seaborn pillow")
    
except Exception as e:
    print(f"❌ Error launching application: {e}")
    import traceback
    traceback.print_exc()
    
    print("\nTroubleshooting:")
    print("1. Make sure all required files are present:")
    print("   - enhanced_air_quality_system.py")
    print("   - fixed_source_predictor.py")
    print("   - source/ folder with CSV files")
    print("2. Install required packages:")
    print("   pip install tensorflow pandas scikit-learn matplotlib seaborn pillow")
    print("3. Check that Python has tkinter support")
    print("4. Try running in administrator mode if permission issues occur")
