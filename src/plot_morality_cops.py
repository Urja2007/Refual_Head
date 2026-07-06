import json
import matplotlib.pyplot as plt
import os
import seaborn as sns

def plot_morality_cops():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results')
    json_path = os.path.join(results_dir, 'level1_top_heads.json')
    
    with open(json_path, 'r') as f:
        data = json.load(f)
        
    top_dfa = data["top_dfa"]
    heads = [f"L{x[0].split('.')[0]}H{x[0].split('.')[1]}" for x in top_dfa]
    scores = [x[1] for x in top_dfa]
    
    # Reverse so highest is at the top of the horizontal bar chart
    heads = heads[::-1]
    scores = scores[::-1]
    
    plt.figure(figsize=(12, 8))
    sns.set_theme(style="whitegrid")
    
    # Create a color palette that gets darker for higher scores
    palette = sns.color_palette("Reds", len(scores))
    
    bars = plt.barh(heads, scores, color=palette)
    
    plt.title('Top 15 "Morality Cops" (Attention Heads writing to Refusal)', fontsize=16, pad=20)
    plt.xlabel('Direct Feature Attribution (DFA) Score', fontsize=14)
    plt.ylabel('Attention Head (Layer.Head)', fontsize=14)
    
    # Add data labels
    for bar in bars:
        width = bar.get_width()
        plt.text(width + 0.002, bar.get_y() + bar.get_height()/2., 
                 f'{width:.4f}', 
                 ha='left', va='center', fontsize=10, fontweight='bold')
                 
    plt.tight_layout()
    plot_path = os.path.join(results_dir, 'morality_cops_plot.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {plot_path}")

if __name__ == "__main__":
    plot_morality_cops()
