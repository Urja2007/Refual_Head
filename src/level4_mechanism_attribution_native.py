import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, get_harmless_prompts, apply_chat_template

def run_level4():
    print("Loading Llama-3-8B-Instruct natively for Level 4 Mechanism Attribution...")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    base_dir = os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r_hat_path = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'models', 'r_hat_level0.pt')
    r_hat = torch.load(r_hat_path).to(torch.float32).to(model.device)
    r_hat_norm = r_hat / torch.norm(r_hat)
    
    # ---------------------------------------------------------
    # Part 1: Extract h_hat (Harmfulness Vector) from Layer 5
    # ---------------------------------------------------------
    print("Extracting h_hat from Layer 5...")
    harmful = get_harmful_prompts(500)
    harmless = get_harmless_prompts(500)
    
    harmful_fmt = [apply_chat_template(tokenizer, p) for p in harmful]
    harmless_fmt = [apply_chat_template(tokenizer, p) for p in harmless]
    
    layer_5_activations = []
    def get_layer5_hook():
        def hook(module, input, output):
            hidden_states = output[0] if isinstance(output, tuple) else output
            last_token_hs = hidden_states[:, -1, :].detach().to(torch.float32)
            layer_5_activations.append(last_token_hs)
        return hook
        
    hook_handle = model.model.layers[5].register_forward_hook(get_layer5_hook())
    
    print("Forward passes for h_hat...")
    with torch.no_grad():
        for batch_prompts in [harmful_fmt, harmless_fmt]:
            for i in tqdm(range(0, len(batch_prompts), 4)):
                batch = batch_prompts[i:i+4]
                inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
                model(**inputs)
                
    hook_handle.remove()
    
    # Calculate means
    all_acts = torch.cat(layer_5_activations, dim=0) # (1000, 4096)
    harmful_mean = all_acts[:500].mean(dim=0)
    harmless_mean = all_acts[500:].mean(dim=0)
    
    h_hat = harmful_mean - harmless_mean
    h_hat_norm = h_hat / torch.norm(h_hat)
    
    torch.save(h_hat_norm, os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'models', 'h_hat_level4.pt'))
    print("Saved h_hat to models/h_hat_level4.pt")
    
    # ---------------------------------------------------------
    # Part 2: Identify Anti-Erasure Neurons in MLPs (Hydra Mechanism)
    # ---------------------------------------------------------
    print("\nScanning MLPs for Anti-Erasure Neurons...")
    # An anti-erasure neuron detects when r_hat is MISSING (negative input weight for r_hat)
    # and WRITES r_hat back into the stream (positive output weight for r_hat)
    
    anti_erasure_neurons = []
    
    for l in tqdm(range(10, model.config.num_hidden_layers)): # Scan layers after Morality Cops
        mlp = model.model.layers[l].mlp
        
        # gate_proj: (intermediate_size, hidden_size)
        W_in = mlp.gate_proj.weight.detach().to(torch.float32) 
        # down_proj: (hidden_size, intermediate_size)
        W_out = mlp.down_proj.weight.detach().to(torch.float32)
        
        # Input correlation with r_hat
        # W_in maps FROM 4096 TO 14336.
        # W_in @ r_hat gives how much each of the 14336 neurons fires when r_hat is present.
        in_corr = torch.matmul(W_in, r_hat_norm) # (14336,)
        
        # Output correlation with r_hat
        # W_out maps FROM 14336 TO 4096.
        # W_out.T @ r_hat gives how much each neuron writes INTO r_hat.
        out_corr = torch.matmul(W_out.T, r_hat_norm) # (14336,)
        
        # Anti-erasure score: strong negative in_corr AND strong positive out_corr
        # We can score them as: -in_corr * out_corr (for neurons where in_corr < 0 and out_corr > 0)
        
        valid_mask = (in_corr < -0.01) & (out_corr > 0.01)
        ae_score = torch.zeros_like(in_corr)
        ae_score[valid_mask] = -in_corr[valid_mask] * out_corr[valid_mask]
        
        # Get top 5 neurons in this layer
        top_scores, top_idx = torch.topk(ae_score, 5)
        
        for i in range(5):
            idx = top_idx[i].item()
            score = top_scores[i].item()
            if score > 0.0001: # Threshold for significant anti-erasure
                anti_erasure_neurons.append({
                    "layer": l,
                    "neuron_idx": idx,
                    "in_corr": in_corr[idx].item(),
                    "out_corr": out_corr[idx].item(),
                    "score": score
                })
                
    # Sort globally by score
    anti_erasure_neurons.sort(key=lambda x: x["score"], reverse=True)
    
    print(f"\nFound {len(anti_erasure_neurons)} strong anti-erasure neurons.")
    for n in anti_erasure_neurons[:10]:
        print(f"Layer {n['layer']} Neuron {n['neuron_idx']}: Score={n['score']:.4f} (In: {n['in_corr']:.4f}, Out: {n['out_corr']:.4f})")
        
    results_dir = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results')
    with open(os.path.join(results_dir, 'level4_anti_erasure.json'), 'w') as f:
        json.dump(anti_erasure_neurons, f, indent=2)

if __name__ == "__main__":
    run_level4()
