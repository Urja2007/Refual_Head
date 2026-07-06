import json
import matplotlib.pyplot as plt
import os

def plot_level2():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    json_path = os.path.join(results_dir, 'level2_core_ablation.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    base_proj = data['baseline']['layer_projections']
    ablated_proj = data['ablated']['layer_projections']
    
    layers = sorted([int(k) for k in base_proj.keys()])
    base_vals = [base_proj[str(l)] for l in layers]
    ablated_vals = [ablated_proj[str(l)] for l in layers]
    
    plt.figure(figsize=(10, 6))
    
    plt.plot(layers, base_vals, label=f'Baseline (Refusal Rate: {data["baseline"]["refusal_rate"]*100:.0f}%)', color='blue', linewidth=2)
    plt.plot(layers, ablated_vals, label=f'Ablated 15 Heads (Refusal Rate: {data["ablated"]["refusal_rate"]*100:.0f}%)', color='red', linestyle='--', linewidth=2)
    
    # Fill between to show the Hydra Effect (where Ablated > Baseline)
    import numpy as np
    base_arr = np.array(base_vals)
    ablated_arr = np.array(ablated_vals)
    plt.fill_between(layers, base_arr, ablated_arr, where=(ablated_arr > base_arr), color='red', alpha=0.3, label='Hydra Effect (Self-Repair)')
    
    plt.title('The Hydra Effect: Residual Stream Projection onto Refusal Vector', fontsize=14)
    plt.xlabel('Transformer Layer', fontsize=12)
    plt.ylabel('Projection onto r_hat (Safety Feature Intensity)', fontsize=12)
    
    plt.axvline(x=11, color='gray', linestyle=':', label='Layer 11 (Optimal r_hat layer)')
    
    plt.legend(fontsize=11)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(layers)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'level2_hydra_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_level2()
