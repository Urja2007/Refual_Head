import os
os.environ["HF_HUB_OFFLINE"] = "1"
print("Importing torch...")
import torch
print("Importing transformers...")
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
print("Tokenizer loaded!")
print("Loading HookedTransformer...")
from transformer_lens import HookedTransformer
model = HookedTransformer.from_pretrained(
    "meta-llama/Meta-Llama-3-8B-Instruct",
    tokenizer=tokenizer,
    device="cuda" if torch.cuda.is_available() else "cpu",
    dtype=torch.float16,
    fold_ln=False,
    center_writing_weights=False,
    center_unembed=False,
    fold_value_biases=False
)
print("Model fully loaded!")
