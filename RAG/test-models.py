import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure API
API_KEY = os.getenv('API_KEY')
if not API_KEY:
    API_KEY = input("Please enter your Google API Key: ")

genai.configure(api_key=API_KEY)
print("✅ API configured successfully!\n")

# List all available models
print("📋 ALL AVAILABLE MODELS:")
print("=" * 60)
try:
    models = genai.list_models()
    for model in models:
        if 'generateContent' in model.supported_generation_methods:
            print(f"Model: {model.name}")
            print(f"  Display Name: {getattr(model, 'display_name', 'N/A')}")
            print(f"  Description: {str(model.description)[:100]}...")
            print()
except Exception as e:
    print(f"Error listing models: {e}")

print("\n🔍 TESTING COMMON MODEL NAMES:")
print("=" * 60)

# Test different model name formats
model_names_to_try = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
    "gemini-1.0-pro",
    "gemini-pro",
    "gemini-1.5-flash-001",
    "gemini-1.5-pro-001",
    "models/gemini-1.5-flash",
    "models/gemini-1.5-pro",
]

for model_name in model_names_to_try:
    try:
        print(f"\nTesting: '{model_name}'")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content("Say 'Hello' in one word")
        print(f"✅ SUCCESS! Response: {response.text}")
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")