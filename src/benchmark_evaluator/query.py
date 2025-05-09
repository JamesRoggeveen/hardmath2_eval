import os
import asyncio
from typing import Tuple
import time
from typing import List

from openai import AsyncOpenAI
from google import genai
from google.genai import types
import anthropic
from tqdm import tqdm
import json
from importlib import resources
from dotenv import load_dotenv
# Load keys from a .env file in your project root
load_dotenv()

# Initialize clients as None
openai_client = None
genai_client = None
anthropic_client = None
deepseek_client = None

def load_config(name: str) -> dict:
    """
    Load benchmark_evaluation/config/{name}.json
    """
    pkg = __package__                  # "benchmark_evaluation"
    resource = f"config/{name}.json"
    with resources.open_text(pkg, resource) as fp:
        return json.load(fp)

config = load_config("model_config")
# Read in all the model configs
SUPPORTED_MODELS_GEMINI = config["SUPPORTED_MODELS_GEMINI"]
SUPPORTED_MODELS_OPENAI = config["SUPPORTED_MODELS_OPENAI"]
SUPPORTED_MODELS_ANTHROPIC = config["SUPPORTED_MODELS_ANTHROPIC"]
SUPPORTED_MODELS_DEEPSEEK = config["SUPPORTED_MODELS_DEEPSEEK"]
SUPPORTED_MODELS = SUPPORTED_MODELS_GEMINI | SUPPORTED_MODELS_OPENAI | SUPPORTED_MODELS_ANTHROPIC | SUPPORTED_MODELS_DEEPSEEK
SYSTEM_INSTRUCTION = config["SYSTEM_INSTRUCTION"]

# Semaphores to limit the number of concurrent requests
openai_sem = asyncio.Semaphore(5)
gemini_sem = asyncio.Semaphore(5)
anthropic_sem = asyncio.Semaphore(5)
deepseek_sem = asyncio.Semaphore(5)

def get_openai_client():
    """Initialize and return OpenAI client if not already initialized."""
    global openai_client
    if openai_client is None:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing OPENAI_API_KEY")
        openai_client = AsyncOpenAI(api_key=api_key)
    return openai_client

def get_gemini_client():
    """Initialize and return Gemini client if not already initialized."""
    global genai_client
    if genai_client is None:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("Missing GEMINI_API_KEY")
        genai_client = genai.Client(api_key=api_key)
    return genai_client

def get_anthropic_client():
    """Initialize and return Anthropic client if not already initialized."""
    global anthropic_client
    if anthropic_client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("Missing ANTHROPIC_API_KEY")
        anthropic_client = anthropic.Anthropic(api_key=api_key)
    return anthropic_client

def get_deepseek_client():
    """Initialize and return DeepSeek client if not already initialized."""
    global deepseek_client
    if deepseek_client is None:
        api_key = os.environ.get("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("Missing DEEPSEEK_API_KEY")
        deepseek_client = AsyncOpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")
    return deepseek_client

async def query_openai_async(client, prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    """Non-blocking OpenAI chat completion."""
    model_id = SUPPORTED_MODELS_OPENAI[model_name]
    print(f"Querying {model_name} with prompt {idx}", flush=True)
    try:
        system_messages = [
            {"role": "system", "content": SYSTEM_INSTRUCTION}
        ]
        response = await client.chat.completions.create(
            model=model_id,
            messages=system_messages + [{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, idx, model_name, False
    except Exception as e:
        return f"Error querying {model_name}: {e}", idx, model_name, True

async def query_gemini_async(client,prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    """Non-blocking Gemini generation via new GenAI SDK."""
    model_id = SUPPORTED_MODELS_GEMINI[model_name]
    print(f"Querying {model_name} with prompt {idx}", flush=True)
    try:
        client = get_gemini_client()
        resp = await client.aio.models.generate_content(
            model=model_id,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION
            ),
            contents=prompt
        )
        return resp.text, idx, model_name, False
    except Exception as e:
        return f"Error querying {model_name}: {e}", idx, model_name, True
    
async def query_anthropic_async(client, prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    """Non-blocking Anthropic chat completion."""
    model_id = SUPPORTED_MODELS_ANTHROPIC[model_name]
    print(f"Querying {model_name} with prompt {idx}", flush=True)
    try:
        response = await client.messages.create(
            model=model_id,
            system=SYSTEM_INSTRUCTION,
            messages=[
                {"role": "user", "content": prompt}
            ],
            max_tokens=8192
        )
        return response.content, idx, model_name, False
    except Exception as e:  
        return f"Error querying {model_name}: {e}", idx, model_name, True

async def query_llm_async(prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    if model_name in SUPPORTED_MODELS_OPENAI:
        client = get_openai_client()
        async with openai_sem:
            return await query_openai_async(client, prompt, model_name, idx)
    elif model_name in SUPPORTED_MODELS_GEMINI:
        client = get_gemini_client()
        async with gemini_sem:
            return await query_gemini_async(client, prompt, model_name, idx)
    elif model_name in SUPPORTED_MODELS_ANTHROPIC:
        client = get_anthropic_client()
        async with anthropic_sem:
            return await query_anthropic_async(client, prompt, model_name, idx)
    elif model_name in SUPPORTED_MODELS_DEEPSEEK:
        client = get_deepseek_client()
        # DeepSeek uses OpenAI's API interface with a custom base URL
        async with deepseek_sem:
            return await query_openai_async(client, prompt, model_name, idx)
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
async def bulk_query_with_progress(prompts: List[str], models: List[str]):
    """Query multiple LLMs with progress bar, then sort by prompt_idx & original model order."""
    pairs = [(i, m) for i, _ in enumerate(prompts) for m in models]
    tasks = [
        asyncio.create_task(query_llm_async(prompts[i], m, i))
        for i, m in pairs
    ]

    results = []
    start = time.perf_counter()
    for task in tqdm(asyncio.as_completed(tasks),
                     total=len(tasks),
                     desc="Querying LLMs"):
        res = await task  # (text, idx, model_name, error)
        results.append(res)
    elapsed = time.perf_counter() - start

    print(f"\n✅ All {len(tasks)} requests completed in {elapsed:.2f}s", flush=True)

    # Build a lookup for model ordering
    model_order = {model: idx for idx, model in enumerate(models)}

    # Sort first by prompt index, then by the order in the 'models' list
    results.sort(key=lambda item: (item[1], model_order[item[2]]))

    return results

async def bulk_query_ordered(prompts, models):
    """Query multiple LLMs without progress bar to maintain order."""
    pairs = [(i, m) for i in range(len(prompts)) for m in models]
    tasks = [asyncio.create_task(query_llm_async(prompts[i], m))
             for i, m in pairs]

    # Gather keeps the original order
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Now zip directly
    return results

if __name__ == "__main__":
    prompts = ["What is 2+2?", "What is 2+3?", "How many r's are there in the word 'strawberry'?", "What is the capital of France?", "What is the capital of Germany?"]
    models = ["GPT-4o-mini", "Gemini 2.0 Flash", "Gemini 2.0 Flash Thinking"]
    results = asyncio.run(bulk_query_with_progress(prompts, models))
    for (text, idx, model, error) in results:
        status = "❌" if error else "✔️"
        print(f"{status} [model={model!r}] prompt_idx={idx} → {text[:60]}{'…' if len(text)>60 else ''}")
