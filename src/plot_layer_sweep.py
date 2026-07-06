import json
import matplotlib.pyplot as plt
import os

def plot_sweep():
    # Load data
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    json_path = os.path.join(results_dir, 'layer_sweep.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    layers = sorted([int(k) for k in data.keys()])
    scores = [data[str(k)] for k in layers]
    
    plt.figure(figsize=(10, 6))
    plt.plot(layers, scores, marker='o', linestyle='-', color='b', linewidth=2, markersize=8)
    
    # Highlight max score
    max_score = max(scores)
    max_layer = layers[scores.index(max_score)]
    
    plt.plot(max_layer, max_score, marker='*', color='red', markersize=15, label=f'Best Layer: {max_layer} ({max_score:.2f})')
    
    # Also highlight Layer 14 (original target)
    if 14 in layers:
        layer_14_score = data['14']
        plt.plot(14, layer_14_score, marker='s', color='green', markersize=10, label=f'Layer 14 ({layer_14_score:.2f})')
    
    plt.title('Separation Score by Layer for Refusal Direction (Llama-3-8B-Instruct)', fontsize=14)
    plt.xlabel('Transformer Layer', fontsize=12)
    plt.ylabel('Separation Score (Harmful vs Harmless)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xticks(layers)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'layer_sweep_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_sweep()
