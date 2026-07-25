import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_token_attribution(words, scores, title="Token Attribution", top_k=20):
    """
    words: list of decoded Vietnamese words 
    scores: attribution scores 
    """
    
    if isinstance(words, list) and isinstance(scores, np.ndarray):
        valid = [(w, s) for w, s in zip(words, scores) if w]  # Bỏ empty strings
    else:
        raise ValueError("words must be list, scores must be np.array")
    
    if len(valid) == 0:
        print("No valid tokens to plot")
        return None
    
    # Sort by absolute value
    valid.sort(key=lambda x: abs(x[1]), reverse=True)
    valid = valid[:top_k]
    
    words_plot, vals = zip(*valid)
    colors = ['#2ecc71' if v > 0 else '#e74c3c' for v in vals]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(words_plot) * 0.25)))
    ax.barh(range(len(words_plot)), vals, color=colors)
    ax.set_yticks(range(len(words_plot)))
    ax.set_yticklabels(words_plot, fontsize=10) 
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Attribution Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#2ecc71', label='Pushes towards AI'),
        Patch(facecolor='#e74c3c', label='Pushes towards Human')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def plot_top_tokens_by_type(top_tokens_per_type, title='Top Token Dự Đoán Là AI'):
    fig, axes = plt.subplots(1, len(top_tokens_per_type), figsize=(10, 6))
    for ax, (gen_type, tokens) in zip(axes, top_tokens_per_type.items()):
        toks, vals = zip(*tokens)
        ax.barh(range(len(toks)), vals, color='#3498db')
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels(toks, fontsize=9)
        ax.invert_yaxis()
        ax.set_title(title, fontweight='bold')
    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
    plt.close(fig)