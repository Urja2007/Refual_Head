import sys
import os
import json
os.environ["HF_HUB_OFFLINE"] = "1"

print("Importing torch...")
import torch
import numpy as np
from tqdm import tqdm

print("Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, get_harmless_prompts, apply_chat_template

def run_level0_native():
    print("Loading Llama-3-8B-Instruct natively...")
    try:
        model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    N_TRAIN = 50
    N_TEST = 20
    print(f"Fetching {N_TRAIN} train / {N_TEST} test samples...")
    harmful_prompts = get_harmful_prompts(N_TRAIN + N_TEST)
    harmless_prompts = get_harmless_prompts(N_TRAIN + N_TEST)

    harmful_train = [apply_chat_template(tokenizer, p) for p in harmful_prompts[:N_TRAIN]]
    harmful_test = [apply_chat_template(tokenizer, p) for p in harmful_prompts[N_TRAIN:]]
    harmless_train = [apply_chat_template(tokenizer, p) for p in harmless_prompts[:N_TRAIN]]
    harmless_test = [apply_chat_template(tokenizer, p) for p in harmless_prompts[N_TRAIN:]]

    def get_residual_stream(prompts, layer_idx):
        # We will use a forward hook to get the output of the specified layer
        residuals = []
        
        def hook_fn(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.dim() == 3:
                last_token_residual = hidden[:, -1, :]
            elif hidden.dim() == 2:
                last_token_residual = hidden[-1, :].unsqueeze(0)
            else:
                last_token_residual = hidden
            residuals.append(last_token_residual.detach().cpu())
            
        # Register hook on the specific layer
        # Llama-3 architecture: model.model.layers[layer_idx]
        hook_handle = model.model.layers[layer_idx].register_forward_hook(hook_fn)
        
        # Process in smaller batches to avoid OOM
        batch_size = 4
        for i in tqdm(range(0, len(prompts), batch_size), desc=f"Processing layer {layer_idx}"):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            with torch.no_grad():
                model(**inputs)
                
        hook_handle.remove()
        
        # Concat all residuals
        return torch.cat(residuals, dim=0)

    # We will sweep layers 10 to 22
    layers_to_sweep = list(range(10, 23))
    best_layer = None
    best_score = -1.0
    best_r_hat = None
    
    print("\n--- Sweeping Layers 10 to 22 ---")
    results = {}
    
    for layer in layers_to_sweep:
        print(f"\nEvaluating Layer {layer}...")
        
        # 1. Compute r_hat on TRAIN set
        harmful_res_train = get_residual_stream(harmful_train, layer)
        harmless_res_train = get_residual_stream(harmless_train, layer)
        
        mean_harmful = harmful_res_train.mean(dim=0)
        mean_harmless = harmless_res_train.mean(dim=0)
        
        r_hat = mean_harmful - mean_harmless
        r_hat = r_hat / torch.norm(r_hat) # Normalize
        
        # 2. Evaluate separation on TEST set
        harmful_res_test = get_residual_stream(harmful_test, layer)
        harmless_res_test = get_residual_stream(harmless_test, layer)
        
        # Project test residuals onto r_hat
        proj_harmful = (harmful_res_test @ r_hat).numpy()
        proj_harmless = (harmless_res_test @ r_hat).numpy()
        
        # Compute separation score (difference in means / pooled std)
        mean_diff = proj_harmful.mean() - proj_harmless.mean()
        pooled_std = np.sqrt((proj_harmful.var() + proj_harmless.var()) / 2.0)
        separation_score = float(mean_diff / (pooled_std + 1e-8))
        
        print(f"Layer {layer} Separation Score: {separation_score:.4f}")
        results[layer] = separation_score
        
        if separation_score > best_score:
            best_score = separation_score
            best_layer = layer
            best_r_hat = r_hat

    print(f"\nBest layer for refusal direction: {best_layer} (Score: {best_score:.4f})")
    
    # Save the best r_hat
    models_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models')
    os.makedirs(models_dir, exist_ok=True)
    save_path = os.path.join(models_dir, 'r_hat_level0.pt')
    torch.save(best_r_hat, save_path)
    print(f"Saved optimal refusal direction to {save_path}")
    
    # Save results to JSON
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results', 'layer_sweep.json'), 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    run_level0_native()
