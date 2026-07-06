import torch
from transformer_lens import HookedTransformer
import sys
import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, get_harmless_prompts, apply_chat_template

def head_contribution_to_refusal(model, harmful_fmt, r_hat, layer, head):
    contributions = []
    # Batch size 1 for simplicity
    for p in harmful_fmt:
        toks = model.to_tokens(p, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=[f'blocks.{layer}.attn.hook_z'])
        # per-head output before W_O
        head_out = cache[f'blocks.{layer}.attn.hook_z'][0, -1, head, :]  # [d_head]
        
        # map through output projection for this head only
        W_O_head = model.blocks[layer].attn.W_O[head]     # [d_head, d_model]
        contribution_vec = head_out @ W_O_head             # [d_model]
        
        # project onto refusal direction
        contributions.append((contribution_vec @ r_hat).item())
        
    return sum(contributions) / len(contributions)

def probe_head(model, harmful_fmt, harmless_fmt, layer, head):
    X, y = [], []
    for p in harmful_fmt:
        toks = model.to_tokens(p, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=[f'blocks.{layer}.attn.hook_z'])
        X.append(cache[f'blocks.{layer}.attn.hook_z'][0, -1, head, :].detach().cpu().numpy())
        y.append(1)
        
    for p in harmless_fmt:
        toks = model.to_tokens(p, prepend_bos=True)
        _, cache = model.run_with_cache(toks, names_filter=[f'blocks.{layer}.attn.hook_z'])
        X.append(cache[f'blocks.{layer}.attn.hook_z'][0, -1, head, :].detach().cpu().numpy())
        y.append(0)
        
    clf = LogisticRegression(max_iter=2000)
    scores = cross_val_score(clf, np.array(X), np.array(y), cv=5)
    return scores.mean()

def run_level1():
    print("Loading Llama-3-8B-Instruct...")
    model = HookedTransformer.from_pretrained(
        "meta-llama/Meta-Llama-3-8B-Instruct",
        device="cuda" if torch.cuda.is_available() else "cpu",
        dtype=torch.float16
    )

    r_hat_path = '../models/r_hat_level0.pt'
    if not os.path.exists(r_hat_path):
        print(f"Error: {r_hat_path} not found. Run Level 0 first.")
        return
    r_hat = torch.load(r_hat_path).to(model.cfg.device)

    N_SAMPLES = 50
    print(f"Fetching {N_SAMPLES} samples for Level 1...")
    harmful = get_harmful_prompts(N_SAMPLES)
    harmless = get_harmless_prompts(N_SAMPLES)
    
    harmful_fmt = apply_chat_template(model, harmful)
    harmless_fmt = apply_chat_template(model, harmless)

    print("Running Method A: Direct Feature Attribution (DFA)...")
    dfa_scores = {}
    for layer in range(model.cfg.n_layers):
        for head in range(model.cfg.n_heads):
            score = head_contribution_to_refusal(model, harmful_fmt, r_hat, layer, head)
            dfa_scores[f"{layer}.{head}"] = score

    top_dfa = sorted(dfa_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top 15 DFA heads: {top_dfa}")

    print("Running Method B: Layer-wise Logistic Regression Probing...")
    # We only probe the top 30 DFA heads to save time, or probe a subset of layers
    candidate_heads = [k for k, v in sorted(dfa_scores.items(), key=lambda x: x[1], reverse=True)[:30]]
    probe_scores = {}
    
    for head_str in candidate_heads:
        layer, head = map(int, head_str.split('.'))
        score = probe_head(model, harmful_fmt, harmless_fmt, layer, head)
        probe_scores[head_str] = score
        
    top_probe = sorted(probe_scores.items(), key=lambda x: x[1], reverse=True)[:15]
    print(f"Top 15 Probe heads: {top_probe}")

    # Intersection
    set_a = set([k for k, v in top_dfa])
    set_b = set([k for k, v in top_probe])
    morality_cops = list(set_a.intersection(set_b))
    print(f"Intersection (Morality Cops): {morality_cops}")

    results = {
        "top_dfa": top_dfa,
        "top_probe": top_probe,
        "morality_cops": morality_cops
    }
    
    os.makedirs('../results', exist_ok=True)
    with open('../results/level1_top_heads.json', 'w') as f:
        json.dump(results, f, indent=2)
    print("Saved results to results/level1_top_heads.json")

if __name__ == "__main__":
    run_level1()
