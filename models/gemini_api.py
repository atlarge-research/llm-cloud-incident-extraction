# import google.generativeai as genai
from google import genai
import yaml

def load_gemini_client():
    # Load API key from config.yaml
    with open(f"./config.yaml", "r") as file:
        config = yaml.safe_load(file)
    gemini_api_key = config["api_keys"]["GEMINI_API_KEY"]
    # genai.configure(api_key=gemini_api_key)

    return genai.Client(api_key=gemini_api_key)


client_gemini = load_gemini_client()