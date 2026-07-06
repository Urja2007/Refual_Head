import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import sys
import os

os.environ["HF_HUB_OFFLINE"] = "1"
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from data_utils import get_harmful_prompts, apply_chat_template

def run_level3():
    print("Loading Llama-3-8B-Instruct natively for Directional Ablation...")
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
    
    # Normalize r_hat for directional ablation
    r_hat_norm = r_hat / torch.norm(r_hat)
    
    active_hooks = []
    
    # We will hook the output of EVERY decoder layer (which includes Attention + MLP)
    def get_directional_ablation_hook():
        def hook(module, input, output):
            # output is a tuple for LlamaDecoderLayer: (hidden_states, ...)
            hidden_states = output[0]
            
            # Project onto r_hat
            # hidden_states: (batch, seq_len, 4096)
            # r_hat_norm: (4096,)
            proj_scalar = torch.matmul(hidden_states, r_hat_norm).unsqueeze(-1)
            
            # Subtract the refusal direction
            ablated_states = hidden_states - (proj_scalar * r_hat_norm.view(1, 1, -1))
            
            # Return new tuple
            if isinstance(output, tuple):
                return (ablated_states,) + output[1:]
            return ablated_states
        return hook
        
    def enable_directional_ablation():
        for l in range(model.config.num_hidden_layers):
            hook = model.model.layers[l].register_forward_hook(get_directional_ablation_hook())
            active_hooks.append(hook)
            
    def disable_ablation():
        for hook in active_hooks:
            hook.remove()
        active_hooks.clear()

    # Get some harmful test prompts
    test_prompts = get_harmful_prompts(100)[50:52] 
    
    for prompt in test_prompts:
        fmt_prompt = apply_chat_template(tokenizer, prompt)
        inputs = tokenizer(fmt_prompt, return_tensors="pt").to(model.device)
        
        gen_kwargs = {
            "max_new_tokens": 100,
            "do_sample": False,
            "pad_token_id": tokenizer.eos_token_id
        }
        
        print(f"\n--- Testing Prompt: {prompt} ---")
        
        # 1. Baseline Generation
        disable_ablation()
        with torch.no_grad():
            base_out = model.generate(**inputs, **gen_kwargs)
        base_text = tokenizer.decode(base_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"BASELINE: {base_text}")
        
        # 2. Full Directional Ablation (Bypassing Hydra Effect)
        enable_directional_ablation()
        with torch.no_grad():
            ablated_out = model.generate(**inputs, **gen_kwargs)
        ablated_text = tokenizer.decode(ablated_out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        print(f"DIRECTIONAL ABLATED: {ablated_text}")
        
    disable_ablation()

if __name__ == "__main__":
    run_level3()
