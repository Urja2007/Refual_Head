import json
import matplotlib.pyplot as plt
import numpy as np
import os

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
results_dir = os.path.join(base_dir, 'results')

with open(os.path.join(results_dir, 'level4_anti_erasure.json'), 'r') as f:
    neurons = json.load(f)
    
top_10 = neurons[:10]

labels = [f"L{n['layer']}.{n['neuron_idx']}" for n in top_10]
in_corr = [n['in_corr'] for n in top_10]
out_corr = [n['out_corr'] for n in top_10]

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(12, 6))
rects1 = ax.bar(x - width/2, in_corr, width, label='Input Correlation (W_in @ r_hat)', color='red')
rects2 = ax.bar(x + width/2, out_corr, width, label='Output Correlation (W_out @ r_hat)', color='green')

ax.set_ylabel('Cosine Correlation')
ax.set_title('Top 10 Anti-Erasure Neurons (Hydra Mechanism)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
ax.grid(axis='y', linestyle='--', alpha=0.7)

fig.tight_layout()
plt.savefig(os.path.join(results_dir, 'level4_anti_erasure_plot.png'))
print(f"Plot saved to {os.path.join(results_dir, 'level4_anti_erasure_plot.png')}")
