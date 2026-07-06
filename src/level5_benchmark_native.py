import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os
import json
from tqdm import tqdm

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, apply_chat_template
from metrics import auto_judge, mcnemar_test, cohens_h, bootstrap_ci

def run_level5():
    print("Loading Llama-3-8B-Instruct natively for Level 5 Benchmarking...")
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    r_hat_path = os.path.join(base_dir, 'models', 'r_hat_level0.pt')
    r_hat = torch.load(r_hat_path).to(torch.float16).to(model.device)
    
    level1_path = os.path.join(base_dir, 'results', 'level1_top_heads.json')
    with open(level1_path, 'r') as f:
        morality_cops = json.load(f)['morality_cops']
        
    num_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // num_heads
    
    # We will test on N_TEST harmful prompts
    N_TEST = 200
    test_prompts = get_harmful_prompts(1000)[700:700+N_TEST] # Use a separate slice!
    fmt_prompts = [apply_chat_template(tokenizer, p) for p in test_prompts]
    
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
            
    active_hooks = []
    
    def get_scaled_hook(heads_dict, scale):
        def hook(module, args):
            hidden_states = args[0]
            batch_size, seq_len, d_model = hidden_states.shape
            reshaped = hidden_states.view(batch_size, seq_len, num_heads, head_dim)
            for h, u in heads_dict.items():
                head_out = reshaped[:, :, h, :]
                proj_scalar = torch.matmul(head_out, u).unsqueeze(-1)
                reshaped[:, :, h, :] = head_out - (scale * proj_scalar * u.view(1, 1, head_dim))
            return (reshaped.view(batch_size, seq_len, d_model),)
        return hook
        
    def enable_scaled_ablation(scale):
        disable_ablation()
        for l, heads_dict in ablation_vectors.items():
            hook = model.model.layers[l].self_attn.o_proj.register_forward_pre_hook(get_scaled_hook(heads_dict, scale))
            active_hooks.append(hook)
            
    def disable_ablation():
        for hook in active_hooks: hook.remove()
        active_hooks.clear()

    scales = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]
    
    results = {}
    detailed_logs = {}
    
    print("\n--- Starting Severity Sweep ---")
    
    baseline_judgements = []
    
    for scale in scales:
        print(f"\n--- Testing Scale: {scale} ---")
        enable_scaled_ablation(scale)
        
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
                
        # Store for comparison
        if scale == 0.0:
            baseline_judgements = judgements
            
        refusal_rate = sum(judgements) / len(judgements)
        
        # Calculate stats vs baseline
        if scale == 0.0:
            p_val, chi2 = 1.0, 0.0
            effect_size = 0.0
        else:
            p_val, chi2 = mcnemar_test(baseline_judgements, judgements)
            effect_size = cohens_h(sum(baseline_judgements)/len(baseline_judgements), refusal_rate)
            
        ci_lower, ci_upper = bootstrap_ci(judgements)
        
        print(f"Refusal Rate: {refusal_rate*100:.1f}% (CI: {ci_lower*100:.1f}% - {ci_upper*100:.1f}%)")
        print(f"Statistical Significance (vs Baseline): p = {p_val:.4e}, Cohen's h = {effect_size:.2f}")
        
        results[str(scale)] = {
            "refusal_rate": refusal_rate,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "mcnemar_p_value": float(p_val),
            "mcnemar_chi2": float(chi2),
            "cohens_h": float(effect_size)
        }
        
    disable_ablation()
    
    results_dir = os.path.join(base_dir, 'results')
    with open(os.path.join(results_dir, 'level5_benchmark_results.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved Level 5 results to results/level5_benchmark_results.json")

if __name__ == "__main__":
    run_level5()
