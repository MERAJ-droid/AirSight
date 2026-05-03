import os
import sys
import warnings
from dotenv import load_dotenv

# Suppress that annoying Google SDK deprecation warning to keep your terminal clean
warnings.filterwarnings("ignore", category=FutureWarning, module="google.generativeai")

import google.generativeai as genai

# Force Python to recognize the 'src' folder
sys.path.append(os.getcwd())
from src.llm.prompts import EXPLAINABILITY_PROMPT

load_dotenv()

# Configure the classic, stable client
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def get_aqi_category(aqi_value):
    if aqi_value <= 50: return "Good"
    elif aqi_value <= 100: return "Moderate"
    elif aqi_value <= 150: return "Unhealthy for Sensitive Groups"
    elif aqi_value <= 200: return "Unhealthy"
    elif aqi_value <= 300: return "Very Unhealthy"
    else: return "Hazardous"

def get_health_advisory(predictions):
    aqi = predictions.get("AQI", 0)
    category = get_aqi_category(aqi)

    formatted_prompt = EXPLAINABILITY_PROMPT.format(
        aqi=round(aqi, 2),
        aqi_category=category,
        pm25=round(predictions.get("PM2.5", 0), 2),
        pm10=round(predictions.get("PM10", 0), 2),
        o3=round(predictions.get("O3", 0), 2),
        co=round(predictions.get("CO", 0), 2),
        so2=round(predictions.get("SO2", 0), 2),
        no2=round(predictions.get("NO2", 0), 2)
    )

    try:
        model = genai.GenerativeModel('gemini-2.5-flash')
        response = model.generate_content(formatted_prompt)
        return response.text
    except Exception as e:
        return f"LLM Error: {str(e)}"

if __name__ == "__main__":
    fake_preds = {"PM2.5": 145.2, "PM10": 180.5, "O3": 45.1, "CO": 2.3, "SO2": 12.4, "NO2": 35.6, "AQI": 198.0}
    print("Testing Classic Gemini Explainer...\n")
    print(get_health_advisory(fake_preds))