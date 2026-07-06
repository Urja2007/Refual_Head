print("Starting script execution...")
import sys
import os
os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
print("Importing transformers to prevent Windows Rust deadlock...")
import transformers
print("Importing torch...")
import torch
print("Importing transformer_lens...")
from transformer_lens import HookedTransformer
print("Importing other modules...")
import json
import numpy as np
from data_utils import get_harmful_prompts, get_harmless_prompts, apply_chat_template

def run_level0():
    print("Loading Llama-3-8B-Instruct...")
    try:
        import sys
        print("Imported sys")
        model = HookedTransformer.from_pretrained(
            "meta-llama/Meta-Llama-3-8B-Instruct",
            device="cuda" if torch.cuda.is_available() else "cpu",
            dtype=torch.float16,
            fold_ln=False,
            center_writing_weights=False,
            center_unembed=False,
            fold_value_biases=False
        )
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return

    N_TRAIN = 50
    N_TEST = 20
    print(f"Fetching {N_TRAIN} train / {N_TEST} test samples...")
    
    harmful_all = get_harmful_prompts(N_TRAIN + N_TEST)
    harmless_all = get_harmless_prompts(N_TRAIN + N_TEST)

    harmful_train_fmt = apply_chat_template(model, harmful_all[:N_TRAIN])
    harmless_train_fmt = apply_chat_template(model, harmless_all[:N_TRAIN])
    
    harmful_test_fmt = apply_chat_template(model, harmful_all[N_TRAIN:])
    harmless_test_fmt = apply_chat_template(model, harmless_all[N_TRAIN:])

    def get_acts(prompts, layer):
        acts = []
        for p in prompts:
            toks = model.to_tokens(p, prepend_bos=True)
            _, cache = model.run_with_cache(toks, names_filter=[f'blocks.{layer}.hook_resid_post'])
            acts.append(cache[f'blocks.{layer}.hook_resid_post'][0, -1].detach().cpu())
        return torch.stack(acts)

    # SWEEP LAYERS to find the most predictive layer
    print("\n--- Sweeping Layers 10 to 22 ---")
    layer_scores = {}
    best_layer = -1
    best_score = -1.0
    best_r_hat = None

    for layer in range(10, 23):
        # 1. Extract r_hat on train set
        h_acts = get_acts(harmful_train_fmt, layer)
        hl_acts = get_acts(harmless_train_fmt, layer)
        r = h_acts.mean(0) - hl_acts.mean(0)
        r_hat = r / r.norm()

        # 2. Evaluate separation on test set
        h_test = get_acts(harmful_test_fmt, layer)
        hl_test = get_acts(harmless_test_fmt, layer)
        
        # Mean projection on r_hat
        h_proj = (h_test @ r_hat).mean().item()
        hl_proj = (hl_test @ r_hat).mean().item()
        
        separation_score = h_proj - hl_proj
        layer_scores[layer] = separation_score
        print(f"Layer {layer}: Separation Score = {separation_score:.4f} (Harmful Proj: {h_proj:.2f}, Harmless Proj: {hl_proj:.2f})")

        if separation_score > best_score:
            best_score = separation_score
            best_layer = layer
            best_r_hat = r_hat

    print(f"\n=> Best Layer Selected: {best_layer} (Score: {best_score:.4f})")
    
    # Save the best r_hat
    os.makedirs('../models', exist_ok=True)
    r_hat_gpu = best_r_hat.to(model.cfg.device)
    torch.save(r_hat_gpu, '../models/r_hat_level0.pt')
    
    # Save the layer scores for verification
    os.makedirs('../results', exist_ok=True)
    with open('../results/level0_layer_sweep.txt', 'w') as f:
        f.write("Layer\tSeparationScore\n")
        for l, s in layer_scores.items():
            f.write(f"{l}\t{s:.4f}\n")
        f.write(f"\nSelected Layer: {best_layer}\n")

    print(f"\n--- Testing Ablation of Layer {best_layer}'s Refusal Direction ---")
    def make_ablation_hook(direction):
        def hook(resid, hook_point):
            proj = (resid @ direction).unsqueeze(-1) * direction
            return resid - proj
        return hook

    hook_fn = make_ablation_hook(r_hat_gpu)
    # Apply hook to all layers from the best_layer onwards
    hook_points = [f'blocks.{l}.hook_resid_post' for l in range(best_layer, model.cfg.n_layers)]

    test_prompt = harmful_all[N_TRAIN]
    test_fmt = harmful_test_fmt[0]
    test_toks = model.to_tokens(test_fmt, prepend_bos=True)

    print("\n[PROMPT]:", test_prompt)
    print("\n[BASELINE OUTPUT] (No Ablation):")
    baseline_out = model.generate(test_toks, max_new_tokens=60, use_past_kv_cache=True)
    print(model.tokenizer.decode(baseline_out[0, test_toks.shape[1]:]))

    print("\n[ABLATED OUTPUT] (Direction Removed):")
    with model.hooks(fwd_hooks=[(hp, hook_fn) for hp in hook_points]):
        ablated_out = model.generate(test_toks, max_new_tokens=60, use_past_kv_cache=False)
        print(model.tokenizer.decode(ablated_out[0, test_toks.shape[1]:]))

if __name__ == "__main__":
    run_level0()
