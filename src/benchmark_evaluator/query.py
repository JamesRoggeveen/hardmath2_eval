import os
import asyncio
from typing import Tuple
import time
from typing import List

from openai import AsyncOpenAI
from google import genai
from tqdm import tqdm

from dotenv import load_dotenv
# Load keys from a .env file in your project root
load_dotenv()


GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
if not GEMINI_API_KEY or not OPENAI_API_KEY:
    raise RuntimeError("Missing GEMINI_API_KEY or OPENAI_API_KEY")

# OpenAI async client (httpx under the hood)
openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)  # :contentReference[oaicite:0]{index=0}

# Gemini new SDK client with async support
genai_client = genai.Client(api_key=GEMINI_API_KEY)

# Semaphores to limit the number of concurrent requests
openai_sem = asyncio.Semaphore(5)
gemini_sem = asyncio.Semaphore(5)

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

async def query_openai_async(prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    """Non-blocking OpenAI chat completion."""
    model_id = SUPPORTED_MODELS_OPENAI[model_name]
    try:
        response = await openai_client.chat.completions.create(
            model=model_id,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content, idx, model_name, False
    except Exception as e:
        return f"Error querying {model_name}: {e}", idx, model_name, True

async def query_gemini_async(prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    """Non-blocking Gemini generation via new GenAI SDK."""
    model_id = SUPPORTED_MODELS_GEMINI[model_name]
    try:
        resp = await genai_client.aio.models.generate_content(
            model=model_id,
            contents=prompt
        )  # :contentReference[oaicite:1]{index=1}
        return resp.text, idx, model_name, False
    except Exception as e:
        return f"Error querying {model_name}: {e}", idx, model_name, True

async def query_llm_async(prompt: str, model_name: str, idx: int = 0) -> Tuple[str, bool]:
    if model_name in SUPPORTED_MODELS_OPENAI:
        async with openai_sem:
            return await query_openai_async(prompt, model_name, idx)
    elif model_name in SUPPORTED_MODELS_GEMINI:
        async with gemini_sem:
            return await query_gemini_async(prompt, model_name, idx)
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

    print(f"\n✅ All {len(tasks)} requests completed in {elapsed:.2f}s")

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
