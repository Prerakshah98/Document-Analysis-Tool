import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables
load_dotenv()

# Get API key safely
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

# Configure Gemini API
genai.configure(api_key=api_key)

print("\nAvailable Gemini Models Supporting generateContent:\n")

try:
    for model in genai.list_models():
        if "generateContent" in model.supported_generation_methods:
            print(f"Model Name : {model.name}")
            print(f"Description: {model.description}")
            print(f"Methods    : {model.supported_generation_methods}")
            print("-" * 60)

except Exception as e:
    print(f"Error: {e}")