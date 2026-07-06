import os
import json
import random

PROCESSED_DIR = r"C:\Users\USER\Desktop\refusal\refusal_direction\dataset\processed"
OUTPUT_DIR = r"C:\Users\USER\Desktop\Urja\Refusal_Head\data"

harmful_files = [
    "advbench.json",
    "harmbench_test.json",
    "harmbench_val.json",
    "jailbreakbench.json",
    "malicious_instruct.json",
    "strongreject.json",
    "tdc2023.json"
]

harmless_file = "alpaca.json"

def combine_harmful():
    combined = []
    for f in harmful_files:
        path = os.path.join(PROCESSED_DIR, f)
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as file:
                data = json.load(file)
                for item in data:
                    if 'instruction' in item and item['instruction']:
                        combined.append(item['instruction'])
    
    # Remove duplicates
    combined = list(set(combined))
    random.shuffle(combined)
    
    out_path = os.path.join(OUTPUT_DIR, 'harmful_prompts.json')
    with open(out_path, 'w', encoding='utf-8') as out_f:
        json.dump(combined, out_f, indent=2)
    print(f"Combined {len(combined)} harmful prompts into {out_path}")
    return combined

def copy_harmless():
    combined = []
    path = os.path.join(PROCESSED_DIR, harmless_file)
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            for item in data:
                if 'instruction' in item and item['instruction']:
                    combined.append(item['instruction'])
                    
    combined = list(set(combined))
    random.shuffle(combined)
    
    out_path = os.path.join(OUTPUT_DIR, 'harmless_prompts.json')
    with open(out_path, 'w', encoding='utf-8') as out_f:
        json.dump(combined, out_f, indent=2)
    print(f"Copied {len(combined)} harmless prompts into {out_path}")
    return combined

if __name__ == "__main__":
    combine_harmful()
    copy_harmless()
