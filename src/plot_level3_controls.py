import json
import matplotlib.pyplot as plt
import os

def plot_level3():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    json_path = os.path.join(results_dir, 'level3_controls.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    labels = list(data.keys())
    rates = [data[k] * 100 for k in labels]
    
    plt.figure(figsize=(10, 6))
    
    colors = ['gray', 'blue', 'purple', 'red']
    bars = plt.bar(labels, rates, color=colors[:len(labels)])
    
    plt.title('Level 3 Control Battery: Refusal Rates', fontsize=14)
    plt.ylabel('Refusal Rate (%)', fontsize=12)
    plt.ylim(0, 105)
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=11)
        
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'level3_controls_plot.png')
    plt.savefig(plot_path, dpi=300)
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_level3()
