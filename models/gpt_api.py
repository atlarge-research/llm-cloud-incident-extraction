import openai
import yaml

def load_openai_client():
    # Load API key from config.yaml, and choose the model
    with open(f"./config.yaml", "r") as file:
        config = yaml.safe_load(file)
    openai_api_key = config["api_keys"]["OPENAI_API_KEY"]

    client_openai = openai.OpenAI(
        api_key=openai_api_key,
        )

    return client_openai

client_openai = load_openai_client()