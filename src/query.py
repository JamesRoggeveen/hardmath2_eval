from openai import OpenAI
import google.generativeai as genai
import os
import asyncio


GEMINI_API_KEY = os.environ.get('GEMINI_API_KEY')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
if not GEMINI_API_KEY:
    raise ValueError("Gemini API key not found in environment variables")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found in environment variables")

# Define supported models
SUPPORTED_MODELS_GEMINI = {
    "Gemini 2.0 Flash Thinking": "gemini-2.0-flash-thinking-exp",
    "Gemini 2.0 Flash": "gemini-2.0-flash",
    "Gemini 2.5 Flash Thinking": "gemini-2.5-pro-exp-03-25",
    "Gemini 2.5 Pro Preview": "gemini-2.5-pro-preview-03-25"
}

SUPPORTED_MODELS_OPENAI = {
    "GPT-4o": "gpt-4o",
    "GPT-4o-mini": "gpt-4o-mini",
    "GPT-o3-mini": "o3-mini",
    "GPT-o1-mini": "o1-mini",
    "GPT-o1": "o1"
}

# Combine the dictionaries using the | operator (Python 3.9+) or dict.update()
SUPPORTED_MODELS = {**SUPPORTED_MODELS_GEMINI, **SUPPORTED_MODELS_OPENAI}

def query_llm(input_string: str, model_name: str) -> Tuple[str, bool]:
    """Query an LLM with the given input string."""
    if model_name in SUPPORTED_MODELS_OPENAI:
        return query_openai(input_string, model_name)
    elif model_name in SUPPORTED_MODELS_GEMINI:
        return query_gemini(input_string, model_name)
    else:
        return f"Unsupported model: {model_name}. Supported models: {', '.join(SUPPORTED_MODELS.keys())}", True

def query_openai(input_string: str, model_name: str) -> Tuple[str, bool]:
    model_id = SUPPORTED_MODELS_OPENAI[model_name]
    
    try:
        client = OpenAI(api_key=OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": input_string}]
        )
        return response.choices[0].message.content, False
    except Exception as e:
        return f"Error querying {model_name}: {str(e)}", True

def query_gemini(input_string: str, model_name: str) -> Tuple[str, bool]:
    model_id = SUPPORTED_MODELS_GEMINI[model_name]
    
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel(f'models/{model_id}')
        response = model.generate_content(input_string,request_options={"timeout": 1000})
        return response.text, False
    except Exception as e:
        return f"Error querying {model_name}: {str(e)}", True