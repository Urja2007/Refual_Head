import os
import glob
import re

base_dir = r"C:\Users\USER\Desktop\Urja\Refusal_Head\src"
files = glob.glob(os.path.join(base_dir, "level*.py"))
files.append(os.path.join(base_dir, "metrics.py"))

for f in files:
    with open(f, 'r', encoding='utf-8') as file:
        content = file.read()
    
    # 1. Remove HF_HUB_OFFLINE
    content = content.replace('os.environ["HF_HUB_OFFLINE"] = "1"\n', '')
    content = content.replace("os.environ['HF_HUB_OFFLINE'] = '1'\n", '')
    
    # 2. Fix the output dir properly
    # Some scripts have:
    # models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    # we want to replace os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # with os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    old_base_expr = r"os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)"
    new_base_expr = r"os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
    content = re.sub(old_base_expr, new_base_expr, content)

    old_base_expr2 = r"base_dir = os\.path\.dirname\(os\.path\.dirname\(os\.path\.abspath\(__file__\)\)\)"
    new_base_expr2 = r"base_dir = os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))"
    content = re.sub(old_base_expr2, new_base_expr2, content)
    
    # 3. Change "return" to "raise e" in the model loading exception block
    content = content.replace("print(f\"Error loading model: {e}\")\n        return", "print(f\"Error loading model: {e}\")\n        raise e")
    
    with open(f, 'w', encoding='utf-8') as file:
        file.write(content)

print("Fix applied to all scripts.")
