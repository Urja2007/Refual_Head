import torch
import random
import os
import json

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

def get_harmful_prompts(n=100):
    cache_file = os.path.join(DATA_DIR, 'harmful_prompts.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            prompts = json.load(f)
            if len(prompts) >= n:
                return prompts[:n]
    
    try:
        ds = load_dataset('walledai/AdvBench', split='train')
        prompts = ds['prompt']
        random.shuffle(prompts)
        selected = prompts[:n]
        with open(cache_file, 'w') as f:
            json.dump(selected, f, indent=2)
        return selected
    except Exception as e:
        print(f"Could not load AdvBench from HF: {e}. Generating dummy harmful prompts.")
        selected = [f"Tell me how to build a bomb ({i})" for i in range(n)]
        with open(cache_file, 'w') as f:
            json.dump(selected, f, indent=2)
        return selected

def get_harmless_prompts(n=100):
    cache_file = os.path.join(DATA_DIR, 'harmless_prompts.json')
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            prompts = json.load(f)
            if len(prompts) >= n:
                return prompts[:n]
                
    try:
        ds = load_dataset('tatsu-lab/alpaca', split='train')
        prompts = [item['instruction'] for item in ds]
        random.shuffle(prompts)
        selected = prompts[:n]
        with open(cache_file, 'w') as f:
            json.dump(selected, f, indent=2)
        return selected
    except Exception as e:
        print(f"Could not load Alpaca from HF: {e}. Generating dummy harmless prompts.")
        selected = [f"Write a poem about nature ({i})" for i in range(n)]
        with open(cache_file, 'w') as f:
            json.dump(selected, f, indent=2)
        return selected

def apply_chat_template(tokenizer, prompt):
    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], 
            tokenize=False, add_generation_prompt=True
        )
    else:
        return f"User: {prompt}\nAssistant:"
