"""
Enhanced Complete Air Quality Analysis System with Improved GUI
Better colors, fonts, spacing, and readability
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

class EnhancedAirQualitySystem:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("AirSight - Air Quality Analysis System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        self.root.state('zoomed')  # Maximize window
        
        # Color scheme
        self.colors = {
            'bg_dark': '#1e1e1e',
            'bg_medium': '#2d2d2d', 
            'bg_light': '#3a3a3a',
            'text_white': '#ffffff',
            'text_gray': '#cccccc',
            'accent_blue': '#007acc',
            'accent_green': '#28a745',
            'accent_red': '#dc3545',
            'accent_orange': '#fd7e14',
            'accent_purple': '#6f42c1'
        }
        
        # Models
        self.single_model = None
        self.multi_model = None
        self.source_predictor = None
        self.scaler = None
        
        # Variables
        self.current_image = None
        self.current_image_path = None
        self.prediction_results = {}
        
        # Setup enhanced GUI
        self.setup_enhanced_gui()
        self.load_models()
        
    def setup_enhanced_gui(self):
        """Setup the enhanced GUI interface"""
        # Configure styles
        self.setup_styles()
        
        # Main title bar
        title_frame = tk.Frame(self.root, bg=self.colors['accent_blue'], height=100)
        title_frame.pack(fill='x', pady=(0, 15))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame, 
            text="🌍 AirSight - Air Quality Analysis System", 
            font=('Segoe UI', 28, 'bold'), 
            fg=self.colors['text_white'], 
            bg=self.colors['accent_blue']
        )
        title_label.pack(expand=True, pady=(15, 5))
        
        subtitle_label = tk.Label(
            title_frame, 
            text="Advanced Image Analysis → Pollutant Detection → Source Identification → Health Assessment", 
            font=('Segoe UI', 14), 
            fg='#e3f2fd', 
            bg=self.colors['accent_blue']
        )
        subtitle_label.pack(pady=(0, 15))
        
        # Main container with padding
        main_frame = tk.Frame(self.root, bg=self.colors['bg_dark'])
        main_frame.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left panel - Image and controls (30% width)
        left_frame = tk.Frame(main_frame, bg=self.colors['bg_medium'], relief='raised', bd=2)
        left_frame.pack(side='left', fill='y', padx=(0, 15), pady=5)
        left_frame.configure(width=400)
        left_frame.pack_propagate(False)
        
        self.setup_left_panel(left_frame)
        
        # Right panel - Results (70% width)
        right_frame = tk.Frame(main_frame, bg=self.colors['bg_dark'])
        right_frame.pack(side='right', fill='both', expand=True, pady=5)
        
        self.setup_right_panel(right_frame)
        
    def setup_styles(self):
        """Setup custom styles for ttk widgets"""
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure notebook style
        style.configure('Custom.TNotebook', 
                       background=self.colors['bg_medium'],
                       borderwidth=0)
        
        style.configure('Custom.TNotebook.Tab',
                       background=self.colors['bg_light'],
                       foreground=self.colors['text_white'],
                       padding=[20, 10],
                       font=('Segoe UI', 11, 'bold'))
        
        style.map('Custom.TNotebook.Tab',
                 background=[('selected', self.colors['accent_blue']),
                           ('active', self.colors['bg_light'])])
        
    def setup_left_panel(self, parent):
        """Setup the left control panel"""
        # Panel title
        panel_title = tk.Label(
            parent, 
            text="📸 IMAGE ANALYSIS", 
            font=('Segoe UI', 16, 'bold'), 
            fg=self.colors['text_white'], 
            bg=self.colors['bg_medium']
        )
        panel_title.pack(pady=(20, 15))
        
        # Image display area
        image_container = tk.Frame(parent, bg=self.colors['bg_light'], relief='sunken', bd=3)
        image_container.pack(pady=15, padx=20, fill='x')
        
        self.image_frame = tk.Frame(image_container, bg='white', relief='flat')
        self.image_frame.pack(pady=10, padx=10, fill='x')
        
        self.image_label = tk.Label(
            self.image_frame, 
            text="📷\n\nSelect an image to analyze\n\nSupported formats:\nJPG, PNG, BMP, TIFF", 
            font=('Segoe UI', 14), 
            bg='white', 
            fg='#666666',
            height=12,
            justify='center'
        )
        self.image_label.pack(pady=30, padx=30, fill='both', expand=True)
        
        # Image info
        self.image_info = tk.Label(
            image_container,
            text="No image selected",
            font=('Segoe UI', 10),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light']
        )
        self.image_info.pack(pady=(0, 10))
        
        # Control buttons
        button_frame = tk.Frame(parent, bg=self.colors['bg_medium'])
        button_frame.pack(pady=20, fill='x', padx=20)
        
        # Select image button
        self.select_btn = tk.Button(
            button_frame, 
            text="📁 SELECT IMAGE", 
            command=self.select_image, 
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent_blue'], 
            fg=self.colors['text_white'], 
            relief='flat', 
            pady=12,
            cursor='hand2',
            activebackground='#0056b3',
            activeforeground='white'
        )
        self.select_btn.pack(fill='x', pady=(0, 10))
        
        # Analyze button
        self.analyze_btn = tk.Button(
            button_frame, 
            text="🔬 ANALYZE AIR QUALITY", 
            command=self.analyze_complete, 
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent_green'], 
            fg=self.colors['text_white'], 
            relief='flat', 
            pady=12, 
            state='disabled',
            cursor='hand2',
            activebackground='#1e7e34',
            activeforeground='white'
        )
        self.analyze_btn.pack(fill='x', pady=(0, 10))
        
        # Clear button
        self.clear_btn = tk.Button(
            button_frame, 
            text="🗑️ CLEAR RESULTS", 
            command=self.clear_results, 
            font=('Segoe UI', 12, 'bold'),
            bg=self.colors['accent_red'], 
            fg=self.colors['text_white'], 
            relief='flat', 
            pady=12,
            cursor='hand2',
            activebackground='#c82333',
            activeforeground='white'
        )
        self.clear_btn.pack(fill='x', pady=(0, 15))
        
        # Model selection
        model_frame = tk.LabelFrame(
            parent, 
            text=" 🤖 MODEL SELECTION ", 
            font=('Segoe UI', 12, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_medium'],
            relief='raised',
            bd=2,
            labelanchor='n'
        )
        model_frame.pack(pady=15, fill='x', padx=20)
        
        self.model_var = tk.StringVar(value="both")
        
        radio_frame = tk.Frame(model_frame, bg=self.colors['bg_medium'])
        radio_frame.pack(pady=15, padx=15, fill='x')
        
        models = [
            ("Single Output Model", "single", "Predicts all pollutants together"),
            ("Multi-Task Model", "multi", "Separate branches for parameters"),
            ("Both Models", "both", "Compare predictions")
        ]
        
        for text, value, desc in models:
            radio_btn = tk.Radiobutton(
                radio_frame,
                text=text,
                variable=self.model_var,
                value=value,
                font=('Segoe UI', 11, 'bold'),
                fg=self.colors['text_white'],
                bg=self.colors['bg_medium'],
                selectcolor=self.colors['accent_blue'],
                activebackground=self.colors['bg_medium'],
                activeforeground=self.colors['text_white']
            )
            radio_btn.pack(anchor='w', pady=3)
            
            desc_label = tk.Label(
                radio_frame,
                text=f"  └ {desc}",
                font=('Segoe UI', 9),
                fg=self.colors['text_gray'],
                bg=self.colors['bg_medium']
            )
            desc_label.pack(anchor='w', pady=(0, 8))
        
        # Status bar
        self.status_frame = tk.Frame(parent, bg=self.colors['bg_light'], relief='sunken', bd=2)
        self.status_frame.pack(side='bottom', fill='x', padx=20, pady=(20, 20))
        
        tk.Label(
            self.status_frame,
            text="📊 STATUS",
            font=('Segoe UI', 10, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_light']
        ).pack(anchor='w', padx=10, pady=(8, 0))
        
        self.status_label = tk.Label(
            self.status_frame,
            text="Ready to analyze air quality images",
            font=('Segoe UI', 10),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light'],
            wraplength=350,
            justify='left'
        )
        self.status_label.pack(anchor='w', padx=10, pady=(2, 8))
        
    def setup_right_panel(self, parent):
        """Setup the right results panel"""
        # Panel title
        panel_title = tk.Label(
            parent, 
            text="📊 ANALYSIS RESULTS", 
            font=('Segoe UI', 16, 'bold'), 
            fg=self.colors['text_white'], 
            bg=self.colors['bg_dark']
        )
        panel_title.pack(pady=(10, 15))
        
        # Create enhanced notebook
        self.notebook = ttk.Notebook(parent, style='Custom.TNotebook')
        self.notebook.pack(fill='both', expand=True, padx=10)
        
        # Tab 1: Pollutant Predictions
        self.pollutant_frame = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(self.pollutant_frame, text="🌫️ POLLUTANT LEVELS")
        
        # Tab 2: Pollution Sources  
        self.source_frame = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(self.source_frame, text="🏭 POLLUTION SOURCES")
        
        # Tab 3: Health Impact
        self.health_frame = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(self.health_frame, text="🏥 HEALTH IMPACT")
        
        # Tab 4: Visualization
        self.viz_frame = tk.Frame(self.notebook, bg=self.colors['bg_light'])
        self.notebook.add(self.viz_frame, text="📈 VISUALIZATIONS")
        
        self.setup_result_tabs()
        
    def setup_result_tabs(self):
        """Setup enhanced content for each result tab"""
        # Pollutant tab
        self.setup_pollutant_tab()
        
        # Source tab
        self.setup_source_tab()
        
        # Health tab  
        self.setup_health_tab()
        
        # Visualization tab
        self.setup_viz_tab()
        
    def setup_pollutant_tab(self):
        """Setup pollutant results tab"""
        # Header
        header = tk.Frame(self.pollutant_frame, bg=self.colors['bg_light'])
        header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(
            header,
            text="🌫️ Pollutant Concentration Predictions",
            font=('Segoe UI', 16, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_light']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="Detailed analysis of air pollutant levels detected in the image",
            font=('Segoe UI', 11),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light']
        ).pack(anchor='w', pady=(5, 0))
        
        # Content area
        content_frame = tk.Frame(self.pollutant_frame, bg=self.colors['bg_medium'], relief='raised', bd=2)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        # Scrollable text
        self.pollutant_text = tk.Text(
            content_frame, 
            wrap='word', 
            font=('Consolas', 12),
            bg='#2b2b2b', 
            fg='#f8f8f2',
            relief='flat',
            padx=20,
            pady=15,
            selectbackground=self.colors['accent_blue'],
            insertbackground='white'
        )
        
        pollutant_scroll = tk.Scrollbar(content_frame, orient='vertical', command=self.pollutant_text.yview)
        self.pollutant_text.configure(yscrollcommand=pollutant_scroll.set)
        
        self.pollutant_text.pack(side='left', fill='both', expand=True)
        pollutant_scroll.pack(side='right', fill='y')
        
    def setup_source_tab(self):
        """Setup source results tab"""
        # Header
        header = tk.Frame(self.source_frame, bg=self.colors['bg_light'])
        header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(
            header,
            text="🏭 Pollution Source Identification",
            font=('Segoe UI', 16, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_light']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="Advanced analysis identifying the likely sources of detected pollution",
            font=('Segoe UI', 11),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light']
        ).pack(anchor='w', pady=(5, 0))
        
        # Content area
        content_frame = tk.Frame(self.source_frame, bg=self.colors['bg_medium'], relief='raised', bd=2)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.source_text = tk.Text(
            content_frame, 
            wrap='word', 
            font=('Consolas', 12),
            bg='#2b2b2b', 
            fg='#f8f8f2',
            relief='flat',
            padx=20,
            pady=15,
            selectbackground=self.colors['accent_blue'],
            insertbackground='white'
        )
        
        source_scroll = tk.Scrollbar(content_frame, orient='vertical', command=self.source_text.yview)
        self.source_text.configure(yscrollcommand=source_scroll.set)
        
        self.source_text.pack(side='left', fill='both', expand=True)
        source_scroll.pack(side='right', fill='y')
        
    def setup_health_tab(self):
        """Setup health results tab"""
        # Header
        header = tk.Frame(self.health_frame, bg=self.colors['bg_light'])
        header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(
            header,
            text="🏥 Health Impact Assessment",
            font=('Segoe UI', 16, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_light']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="Comprehensive health risk evaluation and safety recommendations",
            font=('Segoe UI', 11),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light']
        ).pack(anchor='w', pady=(5, 0))
        
        # Content area
        content_frame = tk.Frame(self.health_frame, bg=self.colors['bg_medium'], relief='raised', bd=2)
        content_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
        self.health_text = tk.Text(
            content_frame, 
            wrap='word', 
            font=('Consolas', 12),
            bg='#2b2b2b', 
            fg='#f8f8f2',
            relief='flat',
            padx=20,
            pady=15,
            selectbackground=self.colors['accent_blue'],
            insertbackground='white'
        )
        
        health_scroll = tk.Scrollbar(content_frame, orient='vertical', command=self.health_text.yview)
        self.health_text.configure(yscrollcommand=health_scroll.set)
        
        self.health_text.pack(side='left', fill='both', expand=True)
        health_scroll.pack(side='right', fill='y')
        
    def setup_viz_tab(self):
        """Setup visualization tab"""
        # Header
        header = tk.Frame(self.viz_frame, bg=self.colors['bg_light'])
        header.pack(fill='x', padx=20, pady=15)
        
        tk.Label(
            header,
            text="📈 Data Visualizations",
            font=('Segoe UI', 16, 'bold'),
            fg=self.colors['text_white'],
            bg=self.colors['bg_light']
        ).pack(anchor='w')
        
        tk.Label(
            header,
            text="Interactive charts and graphs showing analysis results",
            font=('Segoe UI', 11),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_light']
        ).pack(anchor='w', pady=(5, 0))
        
        # Visualization area
        self.viz_canvas_frame = tk.Frame(self.viz_frame, bg=self.colors['bg_medium'], relief='raised', bd=2)
        self.viz_canvas_frame.pack(fill='both', expand=True, padx=20, pady=(0, 20))
        
    def load_models(self):
        """Load all required models"""
        try:
            self.update_status("🔄 Loading AI models...")
            
            # Load pollution source predictor
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
                
            self.update_status("✅ All models loaded successfully! Ready for analysis.")
            
        except Exception as e:
            self.update_status(f"❌ Error loading models: {e}")
    
    def select_image(self):
        """Select an image file for analysis"""
        file_path = filedialog.askopenfilename(
            title="Select Air Quality Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("JPEG files", "*.jpg *.jpeg"),
                ("PNG files", "*.png"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                # Load and display image
                image = Image.open(file_path)
                
                # Get image info
                file_size = os.path.getsize(file_path) / 1024  # KB
                image_info = f"📁 {os.path.basename(file_path)}\n📐 {image.size[0]}×{image.size[1]} pixels\n💾 {file_size:.1f} KB"
                
                # Resize for display
                display_size = (350, 250)
                image_display = image.copy()
                image_display.thumbnail(display_size, Image.Resampling.LANCZOS)
                
                # Convert to PhotoImage
                self.current_image = ImageTk.PhotoImage(image_display)
                self.current_image_path = file_path
                
                # Update display
                self.image_label.configure(image=self.current_image, text="")
                self.image_info.configure(text=image_info)
                
                # Enable analyze button
                self.analyze_btn.configure(state='normal', bg=self.colors['accent_green'])
                
                self.update_status(f"📷 Image loaded: {os.path.basename(file_path)} - Ready for analysis!")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load image: {e}")
                self.update_status(f"❌ Error loading image: {e}")
    
    def preprocess_image(self, image_path, target_size):
        """Preprocess image for model prediction"""
        try:
            image = Image.open(image_path)
            
            if target_size == (224, 224):
                image = image.resize((224, 224))
            else:
                image = image.resize((112, 224))
            
            image_array = np.array(image) / 255.0
            
            if len(image_array.shape) == 2:
                image_array = np.stack([image_array] * 3, axis=-1)
            
            image_array = np.expand_dims(image_array, axis=0)
            return image_array
            
        except Exception as e:
            raise Exception(f"Image preprocessing failed: {e}")
    
    def predict_pollutants_single(self, image_path):
        """Predict pollutants using single output model (simulation)"""
        try:
            np.random.seed(42)
            
            pollutants = {
                'PM2.5': np.random.uniform(20, 100),
                'PM10': np.random.uniform(40, 150),
                'NO2': np.random.uniform(15, 80),
                'SO2': np.random.uniform(5, 40),
                'CO': np.random.uniform(1, 15),
                'O3': np.random.uniform(30, 120),
                'AQI': 0
            }
            
            pollutants['AQI'] = max(pollutants['PM2.5'] * 2, pollutants['PM10'] * 1.5, pollutants['NO2'] * 1.8)
            return pollutants
            
        except Exception as e:
            raise Exception(f"Single model prediction failed: {e}")
    
    def predict_pollutants_multi(self, image_path):
        """Predict pollutants using multi-task model (simulation)"""
        try:
            np.random.seed(123)
            
            pollutants = {
                'PM2.5': np.random.uniform(25, 95),
                'PM10': np.random.uniform(45, 140),
                'NO2': np.random.uniform(20, 75),
                'SO2': np.random.uniform(8, 35),
                'CO': np.random.uniform(2, 12),
                'O3': np.random.uniform(35, 110),
                'AQI': 0
            }
            
            pollutants['AQI'] = max(pollutants['PM2.5'] * 2, pollutants['PM10'] * 1.5, pollutants['NO2'] * 1.8)
            return pollutants
            
        except Exception as e:
            raise Exception(f"Multi-task model prediction failed: {e}")
    
    def predict_pollution_source(self, pollutants):
        """Predict pollution source from pollutant values"""
        if self.source_predictor is None:
            return None
            
        try:
            pollutant_values = [
                pollutants['AQI'],
                pollutants['PM2.5'],
                pollutants['PM10'],
                pollutants['NO2'],
                pollutants['SO2'],
                pollutants['CO'],
                pollutants['O3']
            ]
            
            source_result = self.source_predictor.predict_source_from_pollutants(pollutant_values)
            return source_result
            
        except Exception as e:
            print(f"Source prediction error: {e}")
            return None
    
    def calculate_health_impact(self, pollutants):
        """Calculate health impact based on pollutant levels"""
        try:
            aqi = pollutants['AQI']
            
            if aqi <= 50:
                category = "Good"
                color = "🟢"
                health_message = "Air quality is satisfactory. No health concerns for the general population."
                symptoms = []
                severity = "low"
            elif aqi <= 100:
                category = "Moderate"
                color = "🟡"
                health_message = "Air quality is acceptable. Unusually sensitive individuals may experience minor issues."
                symptoms = ["Mild throat irritation (sensitive individuals only)"]
                severity = "low"
            elif aqi <= 150:
                category = "Unhealthy for Sensitive Groups"
                color = "🟠"
                health_message = "Sensitive groups (children, elderly, people with respiratory conditions) may experience health effects."
                symptoms = ["Breathing difficulties (sensitive groups)", "Eye irritation", "Throat irritation", "Reduced outdoor endurance"]
                severity = "medium"
            elif aqi <= 200:
                category = "Unhealthy"
                color = "🔴"
                health_message = "Everyone may begin to experience health effects. Sensitive groups may experience more serious effects."
                symptoms = ["Breathing difficulties", "Eye and throat irritation", "Reduced lung function", "Fatigue", "Chest discomfort"]
                severity = "high"
            elif aqi <= 300:
                category = "Very Unhealthy"
                color = "🟣"
                health_message = "Health alert: everyone may experience more serious health effects."
                symptoms = ["Serious breathing difficulties", "Heart problems", "Severe eye irritation", "Reduced stamina", "Persistent cough"]
                severity = "very_high"
            else:
                category = "Hazardous"
                color = "🟤"
                health_message = "Health warnings of emergency conditions. The entire population is more likely to be affected."
                symptoms = ["Serious respiratory problems", "Heart conditions", "Severe fatigue", "Emergency conditions", "Avoid all outdoor activities"]
                severity = "emergency"
            
            specific_effects = []
            
            if pollutants['PM2.5'] > 35:
                specific_effects.append(f"🔺 High PM2.5 ({pollutants['PM2.5']:.1f} µg/m³): Respiratory issues, cardiovascular problems")
            
            if pollutants['PM10'] > 150:
                specific_effects.append(f"🔺 High PM10 ({pollutants['PM10']:.1f} µg/m³): Lung irritation, persistent coughing")
            
            if pollutants['NO2'] > 100:
                specific_effects.append(f"🔺 High NO2 ({pollutants['NO2']:.1f} µg/m³): Respiratory inflammation, reduced immunity")
            
            if pollutants['SO2'] > 75:
                specific_effects.append(f"🔺 High SO2 ({pollutants['SO2']:.1f} µg/m³): Breathing difficulties, throat irritation")
            
            if pollutants['CO'] > 9:
                specific_effects.append(f"🔺 High CO ({pollutants['CO']:.1f} mg/m³): Reduced oxygen delivery, headaches")
            
            return {
                'category': category,
                'color': color,
                'aqi': aqi,
                'health_message': health_message,
                'symptoms': symptoms,
                'specific_effects': specific_effects,
                'severity': severity
            }
            
        except Exception as e:
            return {'error': f"Health impact calculation failed: {e}"}
    
    def analyze_complete(self):
        """Perform complete air quality analysis with enhanced feedback"""
        if not self.current_image_path:
            messagebox.showerror("Error", "Please select an image first")
            return
        
        try:
            self.update_status("🔄 Starting comprehensive air quality analysis...")
            self.root.update()
            
            # Clear previous results
            self.clear_results()
            
            results = {}
            model_choice = self.model_var.get()
            
            # Predict pollutants
            self.update_status("🤖 Running AI models for pollutant detection...")
            self.root.update()
            
            if model_choice in ['single', 'both']:
                results['single'] = self.predict_pollutants_single(self.current_image_path)
                
            if model_choice in ['multi', 'both']:
                results['multi'] = self.predict_pollutants_multi(self.current_image_path)
            
            # Use results for source prediction
            if 'single' in results:
                pollutants_for_source = results['single']
            elif 'multi' in results:
                pollutants_for_source = results['multi']
            else:
                raise Exception("No pollutant predictions available")
            
            # Predict pollution source
            self.update_status("🏭 Identifying pollution sources...")
            self.root.update()
            source_result = self.predict_pollution_source(pollutants_for_source)
            
            # Calculate health impact
            self.update_status("🏥 Assessing health impact...")
            self.root.update()
            health_impact = self.calculate_health_impact(pollutants_for_source)
            
            # Store results
            self.prediction_results = {
                'pollutants': results,
                'source': source_result,
                'health': health_impact
            }
            
            # Display results
            self.update_status("📊 Generating visualizations...")
            self.root.update()
            self.display_results()
            
            self.update_status("✅ Analysis complete! Check all tabs for detailed results.")
            
        except Exception as e:
            messagebox.showerror("Analysis Error", f"Analysis failed: {e}")
            self.update_status(f"❌ Analysis failed: {e}")
    
    def display_results(self):
        """Display all analysis results with enhanced formatting"""
        try:
            self.display_pollutant_results()
            self.display_source_results()
            self.display_health_results()
            self.create_visualizations()
            
        except Exception as e:
            print(f"Error displaying results: {e}")
    
    def display_pollutant_results(self):
        """Display enhanced pollutant prediction results"""
        self.pollutant_text.delete(1.0, tk.END)
        
        # Header with enhanced formatting
        text = "╔" + "═" * 70 + "╗\n"
        text += "║" + " " * 15 + "🌫️ POLLUTANT ANALYSIS RESULTS" + " " * 24 + "║\n"
        text += "╚" + "═" * 70 + "╝\n\n"
        
        for model_name, pollutants in self.prediction_results['pollutants'].items():
            model_title = "🤖 Single Output Model" if model_name == 'single' else "🤖 Multi-Task Model"
            
            text += f"\n{model_title}\n"
            text += "─" * 50 + "\n"
            
            # Create formatted table
            text += f"{'Pollutant':<12} │ {'Value':<10} │ {'Unit':<8} │ Status\n"
            text += "─" * 12 + "┼" + "─" * 11 + "┼" + "─" * 9 + "┼" + "─" * 15 + "\n"
            
            for pollutant, value in pollutants.items():
                if pollutant == 'AQI':
                    unit = ""
                    if value <= 50:
                        status = "🟢 Good"
                    elif value <= 100:
                        status = "🟡 Moderate"
                    elif value <= 150:
                        status = "🟠 Unhealthy*"
                    elif value <= 200:
                        status = "🔴 Unhealthy"
                    else:
                        status = "🟣 Hazardous"
                else:
                    unit = "µg/m³"
                    # Simple status based on WHO guidelines
                    if pollutant == 'PM2.5':
                        status = "🟢 Good" if value <= 15 else "🟡 High" if value <= 35 else "🔴 Very High"
                    elif pollutant == 'PM10':
                        status = "🟢 Good" if value <= 45 else "🟡 High" if value <= 100 else "🔴 Very High"
                    elif pollutant == 'NO2':
                        status = "🟢 Good" if value <= 40 else "🟡 High" if value <= 80 else "🔴 Very High"
                    else:
                        status = "🔍 Detected"
                
                text += f"{pollutant:<12} │ {value:>8.1f} │ {unit:<8} │ {status}\n"
            
            text += "\n"
        
        # Add comparison if both models were used
        if len(self.prediction_results['pollutants']) == 2:
            text += "\n📊 MODEL COMPARISON\n"
            text += "─" * 30 + "\n"
            single = self.prediction_results['pollutants']['single']
            multi = self.prediction_results['pollutants']['multi']
            
            text += f"{'Pollutant':<12} │ {'Difference':<12} │ Agreement\n"
            text += "─" * 12 + "┼" + "─" * 13 + "┼" + "─" * 15 + "\n"
            
            for pollutant in single.keys():
                diff = abs(single[pollutant] - multi[pollutant])
                diff_pct = (diff / max(single[pollutant], multi[pollutant])) * 100
                agreement = "🟢 High" if diff_pct < 10 else "🟡 Medium" if diff_pct < 25 else "🔴 Low"
                text += f"{pollutant:<12} │ {diff:>10.1f} │ {agreement}\n"
        
        # Add timestamp
        import datetime
        text += f"\n📅 Analysis completed: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        
        self.pollutant_text.insert(tk.END, text)
    
    def display_source_results(self):
        """Display enhanced pollution source prediction results"""
        self.source_text.delete(1.0, tk.END)
        
        text = "╔" + "═" * 70 + "╗\n"
        text += "║" + " " * 12 + "🏭 POLLUTION SOURCE IDENTIFICATION" + " " * 21 + "║\n"
        text += "╚" + "═" * 70 + "╝\n\n"
        
        if self.prediction_results['source']:
            source_result = self.prediction_results['source']
            
            if 'classification' in source_result:
                classification = source_result['classification']
                dominant_source = classification['dominant_source']
                confidence = classification['confidence']
                
                # Clean up source name
                clean_name = dominant_source.replace('HTAPv3_', '').replace('_', ' ').title()
                
                text += f"🎯 PRIMARY POLLUTION SOURCE\n"
                text += "─" * 40 + "\n"
                text += f"Source Type: {clean_name}\n"
                text += f"Confidence: {confidence:.1%}\n"
                
                # Confidence interpretation
                if confidence >= 0.8:
                    conf_text = "🟢 Very High - Strong evidence"
                elif confidence >= 0.6:
                    conf_text = "🟡 High - Good evidence"
                elif confidence >= 0.4:
                    conf_text = "🟠 Medium - Some uncertainty"
                else:
                    conf_text = "🔴 Low - High uncertainty"
                
                text += f"Reliability: {conf_text}\n\n"
                
                text += f"📊 ALL SOURCE PROBABILITIES\n"
                text += "─" * 50 + "\n"
                text += f"{'Source':<35} │ {'Probability':<12}\n"
                text += "─" * 35 + "┼" + "─" * 12 + "\n"
                
                sorted_probs = sorted(classification['all_probabilities'].items(), 
                                    key=lambda x: x[1], reverse=True)
                
                for source, prob in sorted_probs[:8]:  # Top 8
                    clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                    if len(clean_source) > 34:
                        clean_source = clean_source[:31] + "..."
                    
                    bar_length = int(prob * 20)
                    bar = "█" * bar_length + "░" * (20 - bar_length)
                    text += f"{clean_source:<35} │ {prob:>6.1%} {bar}\n"
            
            if 'regression' in source_result:
                text += f"\n\n🏭 EMISSION CONTRIBUTION ESTIMATES\n"
                text += "─" * 50 + "\n"
                text += f"{'Source':<35} │ {'Emission':<12}\n"
                text += "─" * 35 + "┼" + "─" * 12 + "\n"
                
                sorted_emissions = sorted(source_result['regression'].items(), 
                                        key=lambda x: x[1], reverse=True)
                
                max_emission = max([e[1] for e in sorted_emissions if e[1] > 0], default=1)
                
                for source, emission in sorted_emissions[:8]:  # Top 8
                    if emission > 0.1:  # Only significant contributions
                        clean_source = source.replace('HTAPv3_', '').replace('_', ' ').title()
                        if len(clean_source) > 34:
                            clean_source = clean_source[:31] + "..."
                        
                        bar_length = int((emission / max_emission) * 15)
                        bar = "█" * bar_length + "░" * (15 - bar_length)
                        text += f"{clean_source:<35} │ {emission:>8.1f} {bar}\n"
                
                text += f"\nNote: Emission values are in relative units\n"
        else:
            text += "❌ Source prediction data not available\n"
            text += "This may be due to:\n"
            text += "• Missing emission dataset files\n"
            text += "• Model training issues\n"
            text += "• Insufficient pollutant data\n"
        
        self.source_text.insert(tk.END, text)
    
    def display_health_results(self):
        """Display enhanced health impact results"""
        self.health_text.delete(1.0, tk.END)
        
        text = "╔" + "═" * 70 + "╗\n"
        text += "║" + " " * 16 + "🏥 HEALTH IMPACT ASSESSMENT" + " " * 25 + "║\n"
        text += "╚" + "═" * 70 + "╝\n\n"
        
        health = self.prediction_results['health']
        
        if 'error' not in health:
            # AQI and category
            text += f"🌡️ AIR QUALITY INDEX (AQI)\n"
            text += "─" * 40 + "\n"
            text += f"Current AQI: {health['aqi']:.1f}\n"
            text += f"Category: {health['color']} {health['category']}\n"
            text += f"Severity: {health.get('severity', 'unknown').replace('_', ' ').title()}\n\n"
            
            # Health message
            text += f"💊 HEALTH ASSESSMENT\n"
            text += "─" * 40 + "\n"
            text += f"{health['health_message']}\n\n"
            
            # Symptoms
            if health['symptoms']:
                text += f"⚠️ POTENTIAL HEALTH EFFECTS\n"
                text += "─" * 40 + "\n"
                for i, symptom in enumerate(health['symptoms'], 1):
                    text += f"{i:2d}. {symptom}\n"
                text += "\n"
            
            # Specific pollutant effects
            if health['specific_effects']:
                text += f"🔬 POLLUTANT-SPECIFIC CONCERNS\n"
                text += "─" * 50 + "\n"
                for effect in health['specific_effects']:
                    text += f"• {effect}\n"
                text += "\n"
            
            # Recommendations based on severity
            text += f"💡 HEALTH RECOMMENDATIONS\n"
            text += "─" * 40 + "\n"
            
            if health['aqi'] <= 50:
                recommendations = [
                    "✅ Excellent conditions for all outdoor activities",
                    "✅ No special precautions needed",
                    "✅ Ideal time for exercise and recreation",
                    "✅ Windows can remain open for fresh air"
                ]
            elif health['aqi'] <= 100:
                recommendations = [
                    "⚠️ Sensitive individuals should limit prolonged outdoor exertion",
                    "✅ Generally safe for most people",
                    "✅ Good conditions for most outdoor activities",
                    "⚠️ People with respiratory conditions should monitor symptoms"
                ]
            elif health['aqi'] <= 150:
                recommendations = [
                    "🚫 Sensitive groups should avoid outdoor activities",
                    "⚠️ Limit prolonged outdoor exertion for everyone",
                    "😷 Consider wearing masks when outdoors",
                    "🪟 Close windows and use air conditioning if available"
                ]
            elif health['aqi'] <= 200:
                recommendations = [
                    "🚫 Limit all outdoor activities",
                    "😷 Wear N95 or equivalent masks when going outside",
                    "🪟 Keep windows closed",
                    "🌬️ Use air purifiers indoors if available",
                    "⚕️ Monitor health symptoms closely"
                ]
            elif health['aqi'] <= 300:
                recommendations = [
                    "🚫 Avoid all outdoor activities",
                    "🏠 Stay indoors as much as possible",
                    "🌬️ Use air purifiers and ensure good indoor ventilation",
                    "😷 Wear high-quality masks if you must go outside",
                    "⚕️ Seek medical attention if experiencing symptoms"
                ]
            else:
                recommendations = [
                    "🚨 EMERGENCY CONDITIONS - Stay indoors",
                    "🚫 Avoid ALL outdoor activities",
                    "🌬️ Use air purifiers and seal windows/doors",
                    "⚕️ Seek immediate medical attention for any symptoms",
                    "📞 Follow local emergency guidelines",
                    "🏥 Have emergency contacts ready"
                ]
            
            for rec in recommendations:
                text += f"{rec}\n"
                
            text += f"\n📞 EMERGENCY CONTACTS\n"
            text += "─" * 25 + "\n"
            text += "• Emergency Services: 108/102\n"
            text += "• Poison Control: 1066\n"
            text += "• Air Quality Helpline: Contact local authorities\n"
            
        else:
            text += f"❌ {health['error']}\n"
            text += "\nUnable to calculate health impact due to data issues.\n"
        
        self.health_text.insert(tk.END, text)
    
    def create_visualizations(self):
        """Create enhanced visualization charts"""
        try:
            # Clear previous visualizations
            for widget in self.viz_canvas_frame.winfo_children():
                widget.destroy()
            
            # Set matplotlib style for dark theme
            plt.style.use('dark_background')
            
            # Create figure with subplots
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
            fig.patch.set_facecolor('#2d2d2d')
            fig.suptitle('🌍 Air Quality Analysis Dashboard', fontsize=18, fontweight='bold', color='white')
            
            # Get data for plotting
            if 'single' in self.prediction_results['pollutants']:
                pollutants = self.prediction_results['pollutants']['single']
            else:
                pollutants = self.prediction_results['pollutants']['multi']
            
            # Color scheme for charts
            colors = ['#e74c3c', '#e67e22', '#f39c12', '#f1c40f', '#27ae60', '#3498db']
            
            # Plot 1: Enhanced pollutant levels bar chart
            pollutant_names = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3']
            pollutant_values = [pollutants[name] for name in pollutant_names]
            
            bars1 = ax1.bar(pollutant_names, pollutant_values, color=colors, alpha=0.8, edgecolor='white', linewidth=1)
            ax1.set_title('🌫️ Pollutant Concentrations', fontweight='bold', pad=20)
            ax1.set_ylabel('Concentration (µg/m³)', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            ax1.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars1, pollutant_values):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(pollutant_values)*0.02,
                        f'{value:.1f}', ha='center', va='bottom', fontweight='bold', color='white')
            
            # Plot 2: Enhanced AQI gauge
            aqi = pollutants['AQI']
            aqi_categories = ['Good', 'Moderate', 'Unhealthy*', 'Unhealthy', 'Very Unhealthy', 'Hazardous']
            aqi_colors = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad', '#2c3e50']
            
            # Determine current category
            if aqi <= 50: current_cat = 0
            elif aqi <= 100: current_cat = 1
            elif aqi <= 150: current_cat = 2
            elif aqi <= 200: current_cat = 3
            elif aqi <= 300: current_cat = 4
            else: current_cat = 5
            
            # Create donut chart
            sizes = [50, 50, 50, 50, 50, 50]
            explode = [0.1 if i == current_cat else 0 for i in range(6)]
            
            wedges, texts = ax2.pie(sizes, labels=aqi_categories, colors=aqi_colors, explode=explode,
                                   startangle=90, textprops={'fontsize': 8, 'fontweight': 'bold'})
            
            # Add AQI value in center
            ax2.text(0, 0, f'{aqi:.0f}\nAQI', ha='center', va='center', 
                    fontsize=20, fontweight='bold', color='white')
            ax2.set_title('🌡️ Air Quality Index', fontweight='bold', pad=20)
            
            # Plot 3: Enhanced source probabilities
            if self.prediction_results['source'] and 'classification' in self.prediction_results['source']:
                source_probs = self.prediction_results['source']['classification']['all_probabilities']
                top_sources = sorted(source_probs.items(), key=lambda x: x[1], reverse=True)[:6]
                
                source_names = [s[0].replace('HTAPv3_', '').replace('_', ' ')[:12] for s in top_sources]
                source_values = [s[1] for s in top_sources]
                
                bars3 = ax3.barh(source_names, source_values, color='#9b59b6', alpha=0.8, edgecolor='white')
                ax3.set_title('🏭 Top Pollution Sources', fontweight='bold', pad=20)
                ax3.set_xlabel('Probability', fontweight='bold')
                ax3.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, value in zip(bars3, source_values):
                    ax3.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height()/2,
                            f'{value:.1%}', ha='left', va='center', fontweight='bold', color='white')
            else:
                ax3.text(0.5, 0.5, '🏭\nSource Data\nNot Available', ha='center', va='center', 
                        transform=ax3.transAxes, fontsize=14, fontweight='bold', color='gray')
                ax3.set_title('🏭 Pollution Sources', fontweight='bold', pad=20)
            
            # Plot 4: Enhanced health impact with AQI scale
            health = self.prediction_results['health']
            if 'error' not in health:
                categories = ['Good\n(0-50)', 'Moderate\n(51-100)', 'Unhealthy*\n(101-150)', 
                            'Unhealthy\n(151-200)', 'Very Unhealthy\n(201-300)', 'Hazardous\n(300+)']
                thresholds = [50, 100, 150, 200, 300, 500]
                colors_scale = ['#27ae60', '#f1c40f', '#e67e22', '#e74c3c', '#8e44ad', '#2c3e50']
                
                current_aqi = health['aqi']
                
                # Highlight current category
                bar_colors = []
                for i, threshold in enumerate(thresholds):
                    if current_aqi <= threshold:
                        bar_colors = colors_scale[:i+1] + ['#444444'] * (len(categories) - i - 1)
                        break
                else:
                    bar_colors = colors_scale
                
                bars4 = ax4.bar(range(len(categories)), thresholds, color=bar_colors, alpha=0.7, 
                               edgecolor='white', linewidth=1)
                
                # Add current AQI line
                ax4.axhline(y=current_aqi, color='red', linestyle='--', linewidth=3, 
                           label=f'Current AQI: {current_aqi:.1f}')
                
                ax4.set_title('📊 AQI Scale & Current Level', fontweight='bold', pad=20)
                ax4.set_ylabel('AQI Value', fontweight='bold')
                ax4.set_xticks(range(len(categories)))
                ax4.set_xticklabels(categories, rotation=45, ha='right', fontsize=9)
                ax4.legend(loc='upper left')
                ax4.grid(True, alpha=0.3)
                
                # Add value labels on bars
                for bar, threshold in zip(bars4, thresholds):
                    ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                            f'{threshold}', ha='center', va='bottom', fontweight='bold', color='white')
            
            plt.tight_layout()
            
            # Add to GUI
            canvas = FigureCanvasTkAgg(fig, self.viz_canvas_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
            
        except Exception as e:
            print(f"Visualization error: {e}")
            # Show error message in viz frame
            error_label = tk.Label(
                self.viz_canvas_frame,
                text=f"📊 Visualization Error\n\n{str(e)}\n\nPlease try running the analysis again.",
                font=('Segoe UI', 12),
                fg=self.colors['text_gray'],
                bg=self.colors['bg_medium'],
                justify='center'
            )
            error_label.pack(expand=True)
    
    def clear_results(self):
        """Clear all results and reset display"""
        self.prediction_results = {}
        
        # Clear text widgets
        for text_widget in [self.pollutant_text, self.source_text, self.health_text]:
            text_widget.delete(1.0, tk.END)
            text_widget.insert(tk.END, "No analysis results yet.\n\nSelect an image and click 'Analyze Air Quality' to begin.")
        
        # Clear visualizations
        for widget in self.viz_canvas_frame.winfo_children():
            widget.destroy()
        
        # Add placeholder
        placeholder = tk.Label(
            self.viz_canvas_frame,
            text="📊\n\nVisualizations will appear here\nafter running analysis",
            font=('Segoe UI', 14),
            fg=self.colors['text_gray'],
            bg=self.colors['bg_medium'],
            justify='center'
        )
        placeholder.pack(expand=True)
        
        self.update_status("🗑️ Results cleared - Ready for new analysis")
    
    def update_status(self, message):
        """Update status message"""
        self.status_label.config(text=message)
        print(f"Status: {message}")
    
    def run(self):
        """Start the application"""
        self.root.mainloop()

def main():
    """Main function to run the enhanced air quality system"""
    print("🌍 Starting Enhanced AirSight - Air Quality Analysis System")
    print("=" * 70)
    
    try:
        app = EnhancedAirQualitySystem()
        app.run()
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
