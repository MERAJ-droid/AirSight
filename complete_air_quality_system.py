"""
Complete Air Quality Analysis System
Combines image-to-pollutant prediction with pollution source identification
"""

import os
import sys
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import warnings
warnings.filterwarnings('ignore')

# Import our pollution source predictor
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fixed_source_predictor import PollutionSourcePredictor, load_emission_datasets

class CompleteAirQualitySystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AirSight - Complete Air Quality Analysis System")
        self.root.geometry("1400x900")
        self.root.configure(bg='#f0f0f0')
        
        # Models
        self.single_model = None  # Bengaluru_code.ipynb model
        self.multi_model = None   # Bengaluru_2nd paper_code.ipynb model
        self.source_predictor = None  # Pollution source predictor
        self.scaler = None
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.prediction_results = {}
        
        # Setup GUI
        self.setup_gui()
        self.load_models()
        
    def setup_gui(self):
        """Setup the complete GUI interface"""
        # Main title
        title_frame = tk.Frame(self.root, bg='#2c3e50', height=80)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(title_frame, text="🌍 AirSight - Complete Air Quality Analysis", 
                              font=('Arial', 24, 'bold'), fg='white', bg='#2c3e50')
        title_label.pack(expand=True)
        
        subtitle_label = tk.Label(title_frame, text="Image → Pollutants → Sources → Health Impact", 
                                 font=('Arial', 12), fg='#ecf0f1', bg='#2c3e50')
        subtitle_label.pack()
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#f0f0f0')
        main_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Left panel - Image and controls
        left_frame = tk.LabelFrame(main_frame, text="📸 Image Analysis", font=('Arial', 12, 'bold'), 
                                  bg='#f0f0f0', width=400)
        left_frame.pack(side='left', fill='y', padx=(0, 10))
        left_frame.pack_propagate(False)
        
        # Image display
        self.image_frame = tk.Frame(left_frame, bg='white', relief='sunken', bd=2)
        self.image_frame.pack(pady=10, padx=10, fill='x')
        
        self.image_label = tk.Label(self.image_frame, text="📷\nSelect an image to analyze", 
                                   font=('Arial', 14), bg='white', fg='gray',
                                   width=30, height=15)
        self.image_label.pack(pady=20)
        
        # Buttons
        button_frame = tk.Frame(left_frame, bg='#f0f0f0')
        button_frame.pack(pady=10, fill='x', padx=10)
        
        self.select_btn = tk.Button(button_frame, text="📁 Select Image", 
                                   command=self.select_image, font=('Arial', 11, 'bold'),
                                   bg='#3498db', fg='white', relief='flat', pady=5)
        self.select_btn.pack(fill='x', pady=2)
        
        self.analyze_btn = tk.Button(button_frame, text="🔬 Analyze Air Quality", 
                                    command=self.analyze_complete, font=('Arial', 11, 'bold'),
                                    bg='#e74c3c', fg='white', relief='flat', pady=5, state='disabled')
        self.analyze_btn.pack(fill='x', pady=2)
        
        self.clear_btn = tk.Button(button_frame, text="🗑️ Clear Results", 
                                  command=self.clear_results, font=('Arial', 11, 'bold'),
                                  bg='#95a5a6', fg='white', relief='flat', pady=5)
        self.clear_btn.pack(fill='x', pady=2)
        
        # Model selection
        model_frame = tk.LabelFrame(left_frame, text="🤖 Model Selection", font=('Arial', 10, 'bold'))
        model_frame.pack(pady=10, fill='x', padx=10)
        
        self.model_var = tk.StringVar(value="both")
        tk.Radiobutton(model_frame, text="Single Output Model", variable=self.model_var, 
                      value="single", font=('Arial', 9)).pack(anchor='w')
        tk.Radiobutton(model_frame, text="Multi-Task Model", variable=self.model_var, 
                      value="multi", font=('Arial', 9)).pack(anchor='w')
        tk.Radiobutton(model_frame, text="Both Models", variable=self.model_var, 
                      value="both", font=('Arial', 9)).pack(anchor='w')
        
        # Right panel - Results
        right_frame = tk.Frame(main_frame, bg='#f0f0f0')
        right_frame.pack(side='right', fill='both', expand=True)
        
        # Create notebook for tabbed results
        self.notebook = ttk.Notebook(right_frame)
        self.notebook.pack(fill='both', expand=True)
        
        # Tab 1: Pollutant Predictions
        self.pollutant_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.pollutant_frame, text="🌫️ Pollutant Levels")
        
        # Tab 2: Pollution Sources
        self.source_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.source_frame, text="🏭 Pollution Sources")
        
        # Tab 3: Health Impact
        self.health_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.health_frame, text="🏥 Health Impact")
        
        # Tab 4: Visualization
        self.viz_frame = tk.Frame(self.notebook, bg='white')
        self.notebook.add(self.viz_frame, text="📊 Visualizations")
        
        self.setup_result_tabs()
        
    def setup_result_tabs(self):
        """Setup the content for each result tab"""
        # Pollutant tab
        self.pollutant_text = tk.Text(self.pollutant_frame, wrap='word', font=('Consolas', 10),
                                     bg='#f8f9fa', relief='flat')
        pollutant_scroll = tk.Scrollbar(self.pollutant_frame, orient='vertical', command=self.pollutant_text.yview)
        self.pollutant_text.configure(yscrollcommand=pollutant_scroll.set)
        self.pollutant_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        pollutant_scroll.pack(side='right', fill='y')
        
        # Source tab
        self.source_text = tk.Text(self.source_frame, wrap='word', font=('Consolas', 10),
                                  bg='#f8f9fa', relief='flat')
        source_scroll = tk.Scrollbar(self.source_frame, orient='vertical', command=self.source_text.yview)
        self.source_text.configure(yscrollcommand=source_scroll.set)
        self.source_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        source_scroll.pack(side='right', fill='y')
        
        # Health tab
        self.health_text = tk.Text(self.health_frame, wrap='word', font=('Consolas', 10),
                                  bg='#f8f9fa', relief='flat')
        health_scroll = tk.Scrollbar(self.health_frame, orient='vertical', command=self.health_text.yview)
        self.health_text.configure(yscrollcommand=health_scroll.set)
        self.health_text.pack(side='left', fill='both', expand=True, padx=5, pady=5)
        health_scroll.pack(side='right', fill='y')
        
        # Visualization tab
        self.viz_canvas_frame = tk.Frame(self.viz_frame, bg='white')
        self.viz_canvas_frame.pack(fill='both', expand=True)
        
    def load_models(self):
        """Load all required models"""
        try:
            print("🔄 Loading models...")
            
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
            
            # Try to load image models (if available)
            # Note: In a real implementation, you would load your trained models here
            print("📝 Note: Image-to-pollutant models would be loaded here")
            print("   Add your trained model loading code in load_models() method")
            
            self.update_status("✅ Models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.update_status(f"❌ Error loading models: {e}")
    
    def select_image(self):
        """Select an image file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Air Quality Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                # Resize for display
                display_size = (300, 200)
                image_display = image.copy()
                image_display.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.current_image = ImageTk.PhotoImage(image_display)
                self.current_image_path = file_path
                
                # Update display
                self.image_label.configure(image=self.current_image, text="")
                
                # Enable analyze button
                self.analyze_btn.configure(state='normal')
                
                self.update_status(f"📷 Image loaded: {os.path.basename(file_path)}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
    
    def preprocess_image(self, image_path, target_size):
        """Preprocess image for model prediction"""
        try:
            image = Image.open(image_path)
            
            # Resize image
            if target_size == (224, 224):
                # For single model
                image = image.resize((224, 224))
            else:
                # For multi-task model (224, 112)
                image = image.resize((112, 224))  # PIL uses (width, height)
            
            # Convert to array and normalize
            image_array = np.array(image) / 255.0
            
            # Add batch dimension
            if len(image_array.shape) == 2:  # Grayscale
                image_array = np.stack([image_array] * 3, axis=-1)
            
            image_array = np.expand_dims(image_array, axis=0)
            
            return image_array
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {e}")
    
    def predict_pollutants_single(self, image_path):
        """Predict pollutants using single output model (simulation)"""
        try:
            # Simulate predictions (replace with actual model)
            # This would be your loaded Bengaluru_code.ipynb model
            np.random.seed(42)  # For consistent demo results
            
            pollutants = {
                'PM2.5': np.random.uniform(20, 100),
                'PM10': np.random.uniform(40, 150),
                'NO2': np.random.uniform(15, 80),
                'SO2': np.random.uniform(5, 40),
                'CO': np.random.uniform(1, 15),
                'O3': np.random.uniform(30, 120),
                'AQI': 0  # Will be calculated
            }
            
            # Calculate AQI (simplified)
            pollutants['AQI'] = max(pollutants['PM2.5'] * 2, pollutants['PM10'] * 1.5, pollutants['NO2'] * 1.8)
            
            return pollutants
            
        except Exception as e:
            raise Exception(f"Single model prediction failed: {e}")
    
    def predict_pollutants_multi(self, image_path):
        """Predict pollutants using multi-task model (simulation)"""
        try:
            # Simulate predictions (replace with actual model)
            # This would be your loaded Bengaluru_2nd paper_code.ipynb model
            np.random.seed(123)  # Different seed for variety
            
            pollutants = {
                'PM2.5': np.random.uniform(25, 95),
                'PM10': np.random.uniform(45, 140),
                'NO2': np.random.uniform(20, 75),
                'SO2': np.random.uniform(8, 35),
                'CO': np.random.uniform(2, 12),
                'O3': np.random.uniform(35, 110),
                'AQI': 0  # Will be calculated
            }
            
            # Calculate AQI
            pollutants['AQI'] = max(pollutants['PM2.5'] * 2, pollutants['PM10'] * 1.5, pollutants['NO2'] * 1.8)
            
            return pollutants
            
        except Exception as e:
            raise Exception(f"Multi-task model prediction failed: {e}")
    
    def predict_pollution_source(self, pollutants):
        """Predict pollution source from pollutant values"""
        if self.source_predictor is None:
            return None
            
        try:
            # Format pollutants for source prediction
            pollutant_values = [
                pollutants['AQI'],
                pollutants['PM2.5'],
                pollutants['PM10'],
                pollutants['NO2'],
                pollutants['SO2'],
                pollutants['CO'],
                pollutants['O3']
            ]
            
            # Predict source
            source_result = self.source_predictor.predict_source_from_pollutants(pollutant_values)
            
            return source_result
            
        except Exception as e:
            print(f"Source prediction error: {e}")
            return None
    
    def calculate_health_impact(self, pollutants):
        """Calculate health impact based on pollutant levels"""
        try:
            aqi = pollutants['AQI']
            
            # Health categories based on AQI
            if aqi <= 50:
                category = "Good"
                color = "🟢"
                health_message = "Air quality is satisfactory. No health concerns."
                symptoms = []
            elif aqi <= 100:
                category = "Moderate"
                color = "🟡"
                health_message = "Air quality is acceptable. Sensitive individuals may experience minor issues."
                symptoms = ["Mild throat irritation (sensitive individuals)"]
            elif aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
                color = "🟠"
                health_message = "Sensitive groups may experience health effects."
                symptoms = ["Breathing difficulties (sensitive groups)", "Eye irritation", "Throat irritation"]
            elif aqi <= 200:
                category = "Unhealthy"
                color = "🔴"
                health_message = "Everyone may begin to experience health effects."
                symptoms = ["Breathing difficulties", "Eye and throat irritation", "Reduced lung function", "Fatigue"]
            elif aqi <= 300:
                category = "Very Unhealthy"
                color = "🟣"
                health_message = "Health alert: everyone may experience more serious health effects."
                symptoms = ["Serious breathing difficulties", "Heart problems", "Severe eye irritation", "Reduced stamina"]
            else:
                category = "Hazardous"
                color = "🟤"
                health_message = "Health warnings of emergency conditions."
                symptoms = ["Serious respiratory problems", "Heart conditions", "Severe fatigue", "Avoid outdoor activities"]
            
            # Specific pollutant health effects
            specific_effects = []
            
            if pollutants['PM2.5'] > 35:
                specific_effects.append(f"High PM2.5 ({pollutants['PM2.5']:.1f}): Respiratory issues, heart problems")
            
            if pollutants['PM10'] > 150:
                specific_effects.append(f"High PM10 ({pollutants['PM10']:.1f}): Lung irritation, coughing")
            
            if pollutants['NO2'] > 100:
                specific_effects.append(f"High NO2 ({pollutants['NO2']:.1f}): Respiratory inflammation")
            
            if pollutants['SO2'] > 75:
                specific_effects.append(f"High SO2 ({pollutants['SO2']:.1f}): Breathing difficulties")
            
            if pollutants['CO'] > 9:
                specific_effects.append(f"High CO ({pollutants['CO']:.1f}): Reduced oxygen delivery")
            
            return {
                'category': category,
                'color': color,
                'aqi': aqi,
                'health_message': health_message,
                'symptoms': symptoms,
                'specific_effects': specific_effects
            }
            
        except Exception as e:
            return {'error': f"Health impact calculation failed: {e}"}
    
    def analyze_complete(self):
        """Perform complete air quality analysis"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.update_status("🔄 Analyzing air quality...")
            
            # Clear previous results
            self.clear_results()
            
            results = {}
            model_choice = self.model_var.get()
            
            # Predict pollutants
            if model_choice in ['single', 'both']:
                results['single'] = self.predict_pollutants_single(self.current_image_path)
                
            if model_choice in ['multi', 'both']:
                results['multi'] = self.predict_pollutants_multi(self.current_image_path)
            
            # Use the available results for source prediction
            if 'single' in results:
                pollutants_for_source = results['single']
            elif 'multi' in results:
                pollutants_for_source = results['multi']
            else:
                raise Exception("No pollutant predictions available")
            
            # Predict pollution source
            source_result = self.predict_pollution_source(pollutants_for_source)
            
            # Calculate health impact
            health_impact = self.calculate_health_impact(pollutants_for_source)
            
            # Store results
            self.prediction_results = {
                'pollutants': results,
                'source': source_result,
                'health': health_impact
            }
            
            # Display results
            self.display_results()
            
            self.update_status("✅ Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Analysis failed: {e}")
            self.update_status(f"❌ Analysis failed: {e}")
    
    def display_results(self):
        """Display all analysis results"""
        try:
            # Display pollutant results
            self.display_pollutant_results()
            
            # Display source results
            self.display_source_results()
            
            # Display health results
            self.display_health_results()
            
            # Create visualizations
            self.create_visualizations()
            
        except Exception as e:
            print(f"Error displaying results: {e}")
    
    def display_pollutant_results(self):
        """Display pollutant prediction results"""
        self.pollutant_text.delete(1.0, tk.END)
        
        text = "🌫️ POLLUTANT LEVEL PREDICTIONS\\n"
        text += "=" * 50 + "\\n\\n"
        
        for model_name, pollutants in self.prediction_results['pollutants'].items():
            model_title = "Single Output Model" if model_name == 'single' else "Multi-Task Model"
            text += f"📊 {model_title}\\n"
            text += "-" * 30 + "\\n"
            
            for pollutant, value in pollutants.items():
                if pollutant == 'AQI':
                    text += f"{pollutant:>8}: {value:>8.1f}\\n"
                else:
                    text += f"{pollutant:>8}: {value:>8.1f} µg/m³\\n"
            
            text += "\\n"
        
        # Add comparison if both models were used
        if len(self.prediction_results['pollutants']) == 2:
            text += "📈 MODEL COMPARISON\\n"
            text += "-" * 30 + "\\n"
            single = self.prediction_results['pollutants']['single']
            multi = self.prediction_results['pollutants']['multi']
            
            for pollutant in single.keys():
                diff = abs(single[pollutant] - multi[pollutant])
                text += f"{pollutant:>8}: Δ {diff:>6.1f}\\n"
        
        self.pollutant_text.insert(tk.END, text)
    
    def display_source_results(self):
        """Display pollution source prediction results"""
        self.source_text.delete(1.0, tk.END)
        
        text = "🏭 POLLUTION SOURCE ANALYSIS\\n"
        text += "=" * 50 + "\\n\\n"
        
        if self.prediction_results['source']:
            source_result = self.prediction_results['source']
            
            if 'classification' in source_result:
                classification = source_result['classification']
                dominant_source = classification['dominant_source']
                confidence = classification['confidence']
                
                # Clean up source name
                clean_name = dominant_source.replace('HTAPv3_', '').replace('_', ' ').title()
                
                text += f"🎯 DOMINANT SOURCE\\n"
                text += f"Source: {clean_name}\\n"
                text += f"Confidence: {confidence:.1%}\\n\\n"
                
                text += f"📊 ALL SOURCE PROBABILITIES\\n"
                text += "-" * 30 + "\\n"
                
                sorted_probs = sorted(classification['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                
                for source, prob in sorted_probs[:5]:  # Top 5
                    clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                    text += f"{clean_source:>25}: {prob:>6.1%}\\n"
            
            if 'regression' in source_result:
                text += f"\\n\\n🏭 EMISSION CONTRIBUTIONS\\n"
                text += "-" * 30 + "\\n"
                
                sorted_emissions = sorted(source_result['regression'].items(), 
                                        key=lambda x: x[1], reverse=True)
                
                for source, emission in sorted_emissions[:5]:  # Top 5
                    if emission > 0.1:  # Only significant contributions
                        clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                        text += f"{clean_source:>25}: {emission:>8.1f} units\\n"
        else:
            text += "❌ Source prediction not available\\n"
        
        self.source_text.insert(tk.END, text)
    
    def display_health_results(self):
        """Display health impact results"""
        self.health_text.delete(1.0, tk.END)
        
        text = "🏥 HEALTH IMPACT ASSESSMENT\\n"
        text += "=" * 50 + "\\n\\n"
        
        health = self.prediction_results['health']
        
        if 'error' not in health:
            text += f"🌡️ AIR QUALITY INDEX\\n"
            text += f"AQI: {health['aqi']:.1f}\\n"
            text += f"Category: {health['color']} {health['category']}\\n\\n"
            
            text += f"💊 HEALTH MESSAGE\\n"
            text += f"{health['health_message']}\\n\\n"
            
            if health['symptoms']:
                text += f"⚠️ POTENTIAL SYMPTOMS\\n"
                text += "-" * 25 + "\\n"
                for symptom in health['symptoms']:
                    text += f"• {symptom}\\n"
                text += "\\n"
            
            if health['specific_effects']:
                text += f"🔬 SPECIFIC POLLUTANT EFFECTS\\n"
                text += "-" * 30 + "\\n"
                for effect in health['specific_effects']:
                    text += f"• {effect}\\n"
                text += "\\n"
            
            # Recommendations
            text += f"💡 RECOMMENDATIONS\\n"
            text += "-" * 20 + "\\n"
            
            if health['aqi'] <= 50:
                text += "• Great day for outdoor activities!\\n"
                text += "• No special precautions needed\\n"
            elif health['aqi'] <= 100:
                text += "• Sensitive individuals should limit prolonged outdoor exertion\\n"
                text += "• Generally safe for most people\\n"
            elif health['aqi'] <= 150:
                text += "• Sensitive groups should avoid outdoor activities\\n"
                text += "• Consider wearing masks outdoors\\n"
            elif health['aqi'] <= 200:
                text += "• Limit outdoor activities\\n"
                text += "• Wear N95 masks when going outside\\n"
                text += "• Keep windows closed\\n"
            else:
                text += "• Avoid all outdoor activities\\n"
                text += "• Use air purifiers indoors\\n"
                text += "• Seek medical attention if experiencing symptoms\\n"
        else:
            text += f"❌ {health['error']}\\n"
        
        self.health_text.insert(tk.END, text)
    
    def create_visualizations(self):
        """Create visualization charts"""
        try:
            # Clear previous visualizations
            for widget in self.viz_canvas_frame.winfo_children():
                widget.destroy()
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle('Air Quality Analysis Visualizations', fontsize=16, fontweight='bold')
            
            # Get data for plotting
            if 'single' in self.prediction_results['pollutants']:
                pollutants = self.prediction_results['pollutants']['single']
            else:
                pollutants = self.prediction_results['pollutants']['multi']
            
            # Plot 1: Pollutant levels bar chart
            pollutant_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            pollutant_values = [pollutants[name] for name in pollutant_names]
            
            bars1 = ax1.bar(pollutant_names, pollutant_values, color=['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#27ae60', '#3498db'])
            ax1.set_title('Pollutant Concentrations', fontweight='bold')
            ax1.set_ylabel('Concentration (µg/m³)')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels on bars
            for bar, value in zip(bars1, pollutant_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pollutant_values)*0.01,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 2: AQI gauge
            aqi = pollutants['AQI']
            ax2.pie([aqi, 500-aqi], colors=['#e74c3c' if aqi > 150 else '#f39c12' if aqi > 100 else '#27ae60', '#ecf0f1'],
                   startangle=90, counterclock=False)
            ax2.set_title(f'AQI: {aqi:.1f}', fontweight='bold')
            
            # Plot 3: Source probabilities (if available)
            if self.prediction_results['source'] and 'classification' in self.prediction_results['source']:
                source_probs = self.prediction_results['source']['classification']['all_probabilities']
                top_sources = sorted(source_probs.items(), key=lambda x: x[1], reverse=True)[:5]
                
                source_names = [s[0].replace('HTAPv3_', '').replace('_', ' ')[:15] for s in top_sources]
                source_values = [s[1] for s in top_sources]
                
                bars3 = ax3.barh(source_names, source_values, color='#9b59b6')
                ax3.set_title('Top Pollution Sources', fontweight='bold')
                ax3.set_xlabel('Probability')
                
                # Add value labels
                for bar, value in zip(bars3, source_values):
                    ax3.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                            f'{value:.1%}', ha='left', va='center', fontweight='bold')
            else:
                ax3.text(0.5, 0.5, 'Source data\\nnot available', ha='center', va='center', transform=ax3.transAxes)
                ax3.set_title('Pollution Sources', fontweight='bold')
            
            # Plot 4: Health impact
            health = self.prediction_results['health']
            if 'error' not in health:
                categories = ['Good', 'Moderate', 'Unhealthy\\nSensitive', 'Unhealthy', 'Very\\nUnhealthy', 'Hazardous']
                thresholds = [50, 100, 150, 200, 300, 500]
                colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad', '#2c3e50']
                
                current_aqi = health['aqi']
                bar_colors = []
                
                for i, threshold in enumerate(thresholds):
                    if current_aqi <= threshold:
                        bar_colors.extend([colors[i]] * (i + 1))
                        bar_colors.extend(['#ecf0f1'] * (len(categories) - i - 1))
                        break
                else:
                    bar_colors = colors
                
                bars4 = ax4.bar(range(len(categories)), thresholds, color=bar_colors, alpha=0.7)
                ax4.axhline(y=current_aqi, color='red', linestyle='--', linewidth=2, label=f'Current AQI: {current_aqi:.1f}')
                ax4.set_title('AQI Scale & Current Level', fontweight='bold')
                ax4.set_ylabel('AQI Value')
                ax4.set_xticks(range(len(categories)))
                ax4.set_xticklabels(categories, rotation=45, ha='right')
                ax4.legend()
            
            plt.tight_layout()
            
            # Add to GUI
            canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)
            
        except Exception as e:
            print(f"Visualization error: {e}")
    
    def clear_results(self):
        """Clear all results and reset display"""
        self.prediction_results = {}
        
        # Clear text widgets
        self.pollutant_text.delete(1.0, tk.END)
        self.source_text.delete(1.0, tk.END)
        self.health_text.delete(1.0, tk.END)
        
        # Clear visualizations
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
        
        self.update_status("🗑️ Results cleared")
    
    def update_status(self, message):
        """Update status message"""
        print(f"Status: {message}")
        # You could add a status bar here
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function to run the complete air quality system"""
    print("🌍 Starting AirSight - Complete Air Quality Analysis System")
    print("=" * 60)
    
    try:
        app = CompleteAirQualitySystem()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
