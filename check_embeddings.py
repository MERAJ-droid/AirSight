import os
import warnings
from dotenv import load_dotenv

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

print("Querying Google for available EMBEDDING models for your API Key...\n")

try:
    models = genai.list_models()
    found_any = False
    for m in models:
        if 'embedContent' in m.supported_generation_methods:
            print(f"✅ Found Embedding Model: {m.name}")
            found_any = True
            
    if not found_any:
        print("❌ No embedding models found for this API key.")
        
except Exception as e:
    print(f"Failed to fetch models: {e}")