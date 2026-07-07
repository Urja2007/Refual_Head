import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, get_harmless_prompts, apply_chat_template

def run_level1_native():
    print("Loading Llama-3-8B-Instruct natively...")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    r_hat_path = os.path.join(os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'models', 'r_hat_level0.pt')
    if not os.path.exists(r_hat_path):
        print(f"Error: {r_hat_path} not found. Run Level 0 first.")
        return
        
    r_hat = torch.load(r_hat_path).to(torch.float16).to(model.device)
    
    N_SAMPLES = 1000
    print(f"Fetching {N_SAMPLES} samples for Level 1...")
    harmful_prompts = get_harmful_prompts(N_SAMPLES)
    harmless_prompts = get_harmless_prompts(N_SAMPLES)
    
    harmful_fmt = [apply_chat_template(tokenizer, p) for p in harmful_prompts]
    harmless_fmt = [apply_chat_template(tokenizer, p) for p in harmless_prompts]
    
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # We will hook o_proj in all layers to capture the pre-projection head outputs
    # pre-projection shape: (batch, seq_len, 4096)
    
    head_outputs_harmful = {l: [] for l in range(num_layers)}
    head_outputs_harmless = {l: [] for l in range(num_layers)}
    
    hooks = []
    
    def get_pre_hook(layer_idx, storage_dict):
        def pre_hook(module, input):
            # input[0] shape: (batch, seq_len, 4096)
            # We want the last token: (batch, 4096)
            last_token_input = input[0][:, -1, :].detach().cpu()
            # Reshape to (batch, num_heads, head_dim)
            reshaped = last_token_input.view(-1, num_heads, head_dim)
            storage_dict[layer_idx].append(reshaped)
        return pre_hook
        
    def register_hooks(storage_dict):
        nonlocal hooks
        for h in hooks:
            h.remove()
        hooks = []
        for l in range(num_layers):
            layer_module = model.model.layers[l].self_attn.o_proj
            hook = layer_module.register_forward_pre_hook(get_pre_hook(l, storage_dict))
            hooks.append(hook)

    # Process harmful
    print("Processing harmful prompts...")
    register_hooks(head_outputs_harmful)
    batch_size = 4
    for i in tqdm(range(0, len(harmful_fmt), batch_size)):
        batch = harmful_fmt[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
            
    # Process harmless
    print("Processing harmless prompts...")
    register_hooks(head_outputs_harmless)
    for i in tqdm(range(0, len(harmless_fmt), batch_size)):
        batch = harmless_fmt[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)
            
    for h in hooks:
        h.remove()
        
    # Concatenate batches
    # For each layer, we want a tensor of shape (N_SAMPLES, num_heads, head_dim)
    for l in range(num_layers):
        head_outputs_harmful[l] = torch.cat(head_outputs_harmful[l], dim=0)
        head_outputs_harmless[l] = torch.cat(head_outputs_harmless[l], dim=0)

    print("Running Method A: Direct Feature Attribution (DFA)...")
    dfa_scores = {}
    r_hat_cpu = r_hat.detach().cpu().to(torch.float32)
    
    for l in range(num_layers):
        # o_proj.weight shape is (4096, 4096)
        W_O = model.model.layers[l].self_attn.o_proj.weight.detach().cpu().to(torch.float32) # (d_model, d_model)
        for h in range(num_heads):
            # Extract W_O for this head
            # W_O applies to the input by input @ W_O.T
            # So the slice for head h is W_O.T[h*head_dim : (h+1)*head_dim, :]
            # W_O.T is (4096, 4096)
            W_O_head = W_O.T[h*head_dim : (h+1)*head_dim, :] # (head_dim, d_model)
            
            # DFA = (head_out @ W_O_head) @ r_hat
            # Precompute proj_dir = W_O_head @ r_hat -> (head_dim,)
            proj_dir = W_O_head @ r_hat_cpu
            
            # Get head_outs for harmful prompts
            head_outs = head_outputs_harmful[l][:, h, :].to(torch.float32) # (N_SAMPLES, head_dim)
            
            # Compute attribution
            attributions = head_outs @ proj_dir # (N_SAMPLES,)
            dfa_scores[f"{l}.{h}"] = attributions.mean().item()
            
    top_dfa = sorted(dfa_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top 15 DFA heads: {top_dfa}")
    
    print("Running Method B: Layer-wise Logistic Regression Probing...")
    candidate_heads = [k for k, v in sorted(dfa_scores.items(), key=lambda x: x[1], reverse=True)[:30]]
    probe_scores = {}
    
    for head_str in candidate_heads:
        l, h = map(int, head_str.split('.'))
        
        X_harmful = head_outputs_harmful[l][:, h, :].numpy()
        X_harmless = head_outputs_harmless[l][:, h, :].numpy()
        
        X = np.vstack([X_harmful, X_harmless])
        y = np.array([1]*len(X_harmful) + [0]*len(X_harmless))
        
        clf = LogisticRegression(max_iter=2000)
        scores = cross_val_score(clf, X, y, cv=5)
        probe_scores[head_str] = scores.mean()
        
    top_probe = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top 15 Probe heads: {top_probe}")

    set_a = set([k for k, v in top_dfa])
    set_b = set([k for k, v in top_probe])
    morality_cops = list(set_a.intersection(set_b))
    print(f"Intersection (Morality Cops): {morality_cops}")

    results = {
        "top_dfa": top_dfa,
        "top_probe": top_probe,
        "morality_cops": morality_cops
    }
    
    results_dir = os.path.join(os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'level1_top_heads.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/level1_top_heads.json")

if __name__ == "__main__":
    run_level1_native()
