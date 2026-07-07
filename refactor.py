import os
import glob

base_dir = r"C:\Users\USER\Desktop\Urja\Refusal_Head\src"
files = glob.glob(os.path.join(base_dir, "level*_native.py"))
files.append(os.path.join(base_dir, "metrics.py"))

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Replace hardcoded model
    content = content.replace('"meta-llama/Meta-Llama-3-8B-Instruct"', 'os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")')
    
    # 2. Replace models/ and results/ output directories to point to OUTPUT_DIR
    content = content.replace("os.path.join(base_dir, 'models'", "os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'models'")
    content = content.replace("os.path.join(base_dir, 'results'", "os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results'")
    
    # 3. Handle model.model.layers -> dynamic access
    # Actually, Mistral, Qwen, and Llama all use model.model.layers
    # For num_hidden_layers, Llama uses config.num_hidden_layers, Qwen uses config.num_hidden_layers
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print("Refactor complete.")
