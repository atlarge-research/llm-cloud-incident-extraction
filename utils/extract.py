from models import gpt_api, claude_api, gemini_api
from google.genai import types


# --------------------------
# Model Calls
# --------------------------

def call_gpt(prompt: str, model: str) -> str:
    client_openai = gpt_api.client_openai
    messages = [
        {"role": "system", "content": "You are a system operator help to extract useful information from an incident report."},
        {"role": "user", "content": prompt}
    ]
    response = client_openai.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    response_text = response.choices[0].message.content
    return response, response_text


def call_claude(prompt: str, model: str) -> str:
    client_anthropic = claude_api.client_anthropic
    messages = [{"role": "user", "content": prompt}]
    response = client_anthropic.messages.create(
        model=model,
        max_tokens=4096,
        system = "You are a system operator help to extract useful information from an incident report.",
        messages=messages,
        temperature=0,
    )
    response_text = response.content[0].text
    return response, response_text


def call_gemini_chat(prompt: str, model: str) -> str:
    client_gemini = gemini_api.client_gemini
    model = client_gemini.GenerativeModel(model)
    chat = model.start_chat(
        history=[
            {"role": "model", "parts": "You are a system operator help to extract useful information from an incident report."},
            {"role": "user", "parts": prompt},
        ]
    )
    response = chat.send_message(prompt)
    response_text = response.text
    return response, response_text


def call_gemini(prompt: str, model: str) -> str:
    client_gemini = gemini_api.client_gemini
    response = client_gemini.models.generate_content(
        model = model,
        contents = prompt,
        config = types.GenerateContentConfig(
            # thinking_config=types.ThinkingConfig(thinking_budget=0)
            system_instruction = "You are a system operator help to extract useful information from an incident report.",
            temperature=0,
        ),
    )
    response_text = response.text.strip()
    return response, response_text