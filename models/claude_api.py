import anthropic
import yaml

def load_claude_client():
    # Load API key from config.yaml, and choose the model
    with open(f"./config.yaml", "r") as file:
        config = yaml.safe_load(file)
    anthropic_api_key = config["api_keys"]["ANTHROPIC_API_KEY"]

    client_anthropic = anthropic.Anthropic(
        api_key=anthropic_api_key,
    )
    
    return client_anthropic

client_anthropic = load_claude_client()