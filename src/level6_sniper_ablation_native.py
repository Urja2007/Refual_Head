import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, apply_chat_template
from metrics import auto_judge

def run_level6():
    print("Loading Llama-3-8B-Instruct natively for Level 6 Sniper Ablation...")
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
    
    # ---------------------------------------------------------
    # Target 1: Morality Cops (Attention Heads)
    # ---------------------------------------------------------
    level1_path = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results', 'level1_top_heads.json')
    with open(level1_path, 'r') as f:
        morality_cops = json.load(f)['morality_cops']
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    ablation_vectors = {}
    r_hat_fp32 = r_hat.to(torch.float32)
    for cop in morality_cops:
        l, h = map(int, cop.split('.'))
        W_O = model.model.layers[l].self_attn.o_proj.weight.detach().to(torch.float32)
        W_O_head = W_O.T[h*head_dim : (h+1)*head_dim, :]
        proj_dir = W_O_head @ r_hat_fp32
        norm = torch.norm(proj_dir)
        if norm > 1e-6:
            if l not in ablation_vectors:
                ablation_vectors[l] = {}
            ablation_vectors[l][h] = (proj_dir / norm).to(torch.float16).to(model.device)

    def get_attn_hook(heads_dict):
        def hook(module, args):
            hidden_states = args[0]
            batch_size, seq_len, d_model = hidden_states.shape
            reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
            for h, u in heads_dict.items():
                head_out = reshaped[:, :, h, :]
                proj_scalar = torch.matmul(head_out, u).unsqueeze(-1)
                reshaped[:, :, h, :] = head_out - (proj_scalar * u.view(1, 1, head_dim))
            return (reshaped.view(batch_size, seq_len, d_model),)
        return hook

    # ---------------------------------------------------------
    # Target 2: Anti-Erasure Neurons (MLPs)
    # ---------------------------------------------------------
    level4_path = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results', 'level4_anti_erasure.json')
    with open(level4_path, 'r') as f:
        anti_erasure_neurons = json.load(f)
        
    # Take top 50 neurons
    top_50_ae = anti_erasure_neurons[:50]
    ae_by_layer = {}
    for n in top_50_ae:
        l = n['layer']
        idx = n['neuron_idx']
        if l not in ae_by_layer:
            ae_by_layer[l] = []
        ae_by_layer[l].append(idx)

    def get_mlp_hook(neuron_indices):
        def hook(module, args):
            # The input to mlp.down_proj is (batch, seq_len, intermediate_size)
            hidden_states = args[0]
            # Zero out the specific neurons
            for idx in neuron_indices:
                hidden_states[:, :, idx] = 0.0
            return (hidden_states,)
        return hook

    # ---------------------------------------------------------
    # Setup Hooks
    # ---------------------------------------------------------
    active_hooks = []
    def disable_ablation():
        for hook in active_hooks: hook.remove()
        active_hooks.clear()

    def enable_sniper_ablation():
        disable_ablation()
        # 1. Hook Morality Cops
        for l, heads_dict in ablation_vectors.items():
            hook = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(get_attn_hook(heads_dict))
            active_hooks.append(hook)
        
        # 2. Hook Anti-Erasure Neurons
        for l, indices in ae_by_layer.items():
            hook = model.model.layers[l].mlp.down_proj.register_forward_pre_hook(get_mlp_hook(indices))
            active_hooks.append(hook)

    # ---------------------------------------------------------
    # Generation & Evaluation
    # ---------------------------------------------------------
    N_TEST = 200
    # Use a specific slice to ensure no overlap
    test_prompts = get_harmful_prompts(1000)[800:800+N_TEST] 
    fmt_prompts = [apply_chat_template(tokenizer, p) for p in test_prompts]
    
    print(f"\n--- Running Sniper Ablation (15 Cops + 50 AE Neurons) on {N_TEST} Prompts ---")
    
    # We will test Baseline, Core Ablation Only, and Sniper Ablation
    modes = {
        "baseline": lambda: disable_ablation(),
        "core_only": lambda: [disable_ablation(), [active_hooks.append(model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(get_attn_hook(hd))) for l, hd in ablation_vectors.items()]],
        "sniper_dual": enable_sniper_ablation
    }
    
    results = {}
    generations_log = {k: [] for k in modes.keys()}
    
    for mode_name, setup_fn in modes.items():
        print(f"\nEvaluating Mode: {mode_name}")
        setup_fn()
        
        judgements = []
        batch_size = 4
        for i in tqdm(range(0, len(fmt_prompts), batch_size)):
            batch = fmt_prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)
            
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=50, do_sample=False, pad_token_id=tokenizer.eos_token_id)
                
            for j, out_tokens in enumerate(out):
                gen_text = tokenizer.decode(out_tokens[inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
                judgement = auto_judge(gen_text)
                judgements.append(judgement)
                generations_log[mode_name].append({
                    "prompt": test_prompts[i+j],
                    "generation": gen_text,
                    "refused": judgement == 1
                })
                
        refusal_rate = sum(judgements) / len(judgements)
        results[mode_name] = {"refusal_rate": refusal_rate}
        print(f"{mode_name} Refusal Rate: {refusal_rate*100:.1f}%")
        
    disable_ablation()
    
    results_dir = os.path.join(os.environ.get('OUTPUT_DIR', base_dir), 'results')
    with open(os.path.join(results_dir, 'level6_sniper_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    with open(os.path.join(results_dir, 'level6_sniper_generations.json'), 'w') as f:
        json.dump(generations_log, f, indent=2)
        
    print("\nSaved Level 6 Sniper results to results/")

if __name__ == "__main__":
    run_level6()
