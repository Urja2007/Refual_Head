import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json
from collections import defaultdict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, apply_chat_template

def run_level2_native():
    print("Loading Llama-3-8B-Instruct natively for Generation...")
    tokenizer = AutoTokenizer.from_pretrained(os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"))
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        os.environ.get("TARGET_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct"),
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    base_dir = os.environ.get('OUTPUT_DIR', os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    r_hat_path = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'models', 'r_hat_level0.pt')
    r_hat = torch.load(r_hat_path).to(torch.float16).to(model.device)
    
    level1_path = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results', 'level1_top_heads.json')
    with open(level1_path, 'r') as f:
        level1_data = json.load(f)
        
    morality_cops = level1_data['morality_cops']
    print(f"Loaded {len(morality_cops)} Morality Cops for orthogonal ablation.")
    
    # Precompute ablation vectors `u` for each Morality Cop
    # organized by layer: {layer_idx: {head_idx: u_vector}}
    ablation_vectors = defaultdict(dict)
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    r_hat_fp32 = r_hat.to(torch.float32)
    
    for cop in morality_cops:
        l, h = map(int, cop.split('.'))
        
        W_O = model.model.layers[l].self_attn.o_proj.weight.detach().to(torch.float32)
        W_O_head = W_O.T[h*head_dim : (h+1)*head_dim, :]
        
        # Direction in head space that writes into r_hat
        proj_dir = W_O_head @ r_hat_fp32 # (head_dim,)
        
        # Normalize to get unit vector u
        norm = torch.norm(proj_dir)
        if norm > 1e-6:
            u = (proj_dir / norm).to(torch.float16).to(model.device)
            ablation_vectors[l][h] = u
            
    # Function to register ablation hooks
    active_hooks = []
    
    def get_ablation_hook(layer_idx, heads_u_dict):
        def pre_hook(module, args):
            # args[0] shape: (batch, seq_len, 4096)
            hidden_states = args[0]
            batch_size, seq_len, d_model = hidden_states.shape
            
            # Reshape to easily isolate heads
            reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
            
            for h, u in heads_u_dict.items():
                head_out = reshaped[:, :, h, :]
                
                # Project head_out onto u
                # u is shape (head_dim,)
                # scalar projection: (batch, seq_len)
                proj_scalar = torch.matmul(head_out, u).unsqueeze(-1) # (batch, seq_len, 1)
                
                # Orthogonalize: subtract the projection component
                ablated_out = head_out - (proj_scalar * u.view(1, 1, head_dim))
                
                # Overwrite head output
                reshaped[:, :, h, :] = ablated_out
                
            # Reshape back and return as new input
            new_hidden_states = reshaped.view(batch_size, seq_len, d_model)
            return (new_hidden_states,)
        return pre_hook
        
    def enable_ablation():
        disable_ablation() # Ensure clean state
        for l, heads_u_dict in ablation_vectors.items():
            layer_module = model.model.layers[l].self_attn.o_proj
            hook = layer_module.register_forward_pre_hook(get_ablation_hook(l, heads_u_dict))
            active_hooks.append(hook)
            
    def disable_ablation():
        for hook in active_hooks:
            hook.remove()
        active_hooks.clear()

    # Get some harmful test prompts
    test_prompts = get_harmful_prompts(100)[50:55] # Use 5 from the test set
    
    results = []
    
    for prompt in test_prompts:
        fmt_prompt = apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(fmt_prompt, return_tensors="pt").to(model.device)
        
        # Generation config
        gen_kwargs = {
            "max_new_tokens": 100,
            "do_sample": False, # Greedy for reproducibility
            "pad_token_id": tokenizer.eos_token_id
        }
        
        print(f"\n--- Testing Prompt: {prompt} ---")
        
        # 1. Baseline Generation (No Ablation)
        disable_ablation()
        with torch.no_grad():
            base_out = model.generate(**inputs, **gen_kwargs)
        base_text = tokenizer.decode(base_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"BASELINE: {base_text}")
        
        # 2. Ablated Generation (Morality Cops Surgically Removed)
        enable_ablation()
        with torch.no_grad():
            ablated_out = model.generate(**inputs, **gen_kwargs)
        ablated_text = tokenizer.decode(ablated_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"ABLATED: {ablated_text}")
        
        results.append({
            "prompt": prompt,
            "baseline": base_text,
            "ablated": ablated_text
        })
        
    disable_ablation()
    
    # Save results
    results_dir = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results')
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, 'level2_ablation_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved qualitative results to results/level2_ablation_results.json")

if __name__ == "__main__":
    run_level2_native()
