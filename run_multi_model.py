import os
import subprocess
import time

MODELS_TO_TEST = [
    "meta-llama/Llama-2-7b-chat-hf",
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.1",
    "mistralai/Mistral-7B-Instruct-v0.2",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "mistralai/Ministral-3B-Instruct",
    "Qwen/Qwen1.5-7B-Chat",
    "Qwen/Qwen2-7B-Instruct",
    "Qwen/Qwen2-1.5B-Instruct",
    "Qwen/Qwen2-0.5B-Instruct"
]

base_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = base_dir

scripts_to_run = [
    "src/level0_sanity_check_native.py",
    "src/level1_head_identification_native.py",
    "src/level2_core_ablation_native.py",
    "src/level3_control_battery_native.py",
    "src/level4_mechanism_attribution_native.py",
    "src/level5_benchmark_native.py",
    "src/level6_sniper_ablation_native.py"
]

for model in MODELS_TO_TEST:
    print(f"\n=========================================================")
    print(f"STARTING MULTI-MODEL SWEEP FOR: {model}")
    print(f"=========================================================\n")
    
    # Create output directory for this specific model
    safe_name = model.replace("/", "_")
    output_dir = os.path.join(root_dir, f"output_{safe_name}")
    os.makedirs(os.path.join(output_dir, "models"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "results"), exist_ok=True)
    
    # Set Environment Variables
    env = os.environ.copy()
    env["TARGET_MODEL"] = model
    env["OUTPUT_DIR"] = output_dir
    
    success = True
    for script in scripts_to_run:
        script_path = os.path.join(root_dir, script)
        if not os.path.exists(script_path):
            print(f"Skipping {script}, not found.")
            continue
            
        print(f"--> Running {script} ...")
        start_time = time.time()
        
        try:
            # Run the script and stream output
            result = subprocess.run(
                ["python", script_path],
                env=env,
                cwd=root_dir,
                check=True
            )
            print(f"--> Finished {script} in {time.time() - start_time:.1f}s")
        except subprocess.CalledProcessError as e:
            print(f"!!! Error running {script} on {model} !!!")
            success = False
            break # Stop pipeline for this model and move to the next
            
    if success:
        print(f"\nSuccessfully completed all levels for {model}! Results are in {output_dir}\n")
    else:
        print(f"\nPipeline failed for {model}. Moving to next model...\n")

print("\nALL MULTI-MODEL SWEEPS COMPLETED!")
