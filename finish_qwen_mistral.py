import os
import subprocess
import time

base_dir = os.path.dirname(os.path.abspath(__file__))

# FINISH QWEN MODELS
qwen_models = [
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct"
]

qwen_scripts = [
    "src/level3_control_battery_qwen.py",
    "src/level4_mechanism_attribution_native.py",
    "src/level5_benchmark_native.py",
    "src/level6_sniper_ablation_native.py"
]

for model in qwen_models:
    print(f"\n[FINISHING PIPELINE FOR: {model}]")
    safe_name = model.replace("/", "_")
    output_dir = os.path.join(base_dir, f"output_{safe_name}")
    
    env = os.environ.copy()
    env["TARGET_MODEL"] = model
    env["OUTPUT_DIR"] = output_dir
    
    for script in qwen_scripts:
        script_path = os.path.join(base_dir, script)
        print(f"--> Running {script} ...")
        try:
            subprocess.run(["python", script_path], env=env, cwd=base_dir, check=True)
        except subprocess.CalledProcessError:
            print(f"!!! Error running {script} on {model} !!!")
            break

# FINISH MISTRAL MODEL
mistral_model = "mistralai/Mistral-7B-Instruct-v0.1"
print(f"\n[FINISHING PIPELINE FOR: {mistral_model}]")
safe_name = mistral_model.replace("/", "_")
output_dir = os.path.join(base_dir, f"output_{safe_name}")

env = os.environ.copy()
env["TARGET_MODEL"] = mistral_model
env["OUTPUT_DIR"] = output_dir

mistral_scripts = [
    "src/level5_benchmark_mistral.py",
    "src/level6_sniper_ablation_native.py"
]

for script in mistral_scripts:
    script_path = os.path.join(base_dir, script)
    print(f"--> Running {script} ...")
    try:
        subprocess.run(["python", script_path], env=env, cwd=base_dir, check=True)
    except subprocess.CalledProcessError:
        print(f"!!! Error running {script} on {mistral_model} !!!")
        break

print("\nDONE!")
