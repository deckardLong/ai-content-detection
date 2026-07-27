import re
import numpy as np
import html as html_lib
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from IPython.display import HTML, display

GREEN = '#2ecc71'
RED = '#e74c3c'

PUNCT_RE = re.compile(r'^[\W_]+$')

def filter_punct(words, scores):
    keep = [(w, s) for w, s in zip(words, scores) if not PUNCT_RE.match(w)]
    return zip(*keep) if keep else ([], [])

def plot_token_attribution(words, scores, title="Token Attribution", top_k=20):
    """
    words: list of decoded Vietnamese words 
    scores: attribution scores 
    """
    
    if isinstance(words, list) and isinstance(scores, np.ndarray):
        valid = [(w, s) for w, s in zip(words, scores) if w]  # Drop empty strings
    else:
        raise ValueError("words must be list, scores must be np.array")
    
    if len(valid) == 0:
        print("No valid tokens to plot")
        return None
    
    # Sort by absolute value
    valid.sort(key=lambda x: abs(x[1]), reverse=True)
    valid = valid[:top_k]
    
    words_plot, vals = zip(*valid)
    colors = [GREEN if v > 0 else RED for v in vals]
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(words_plot) * 0.25)))
    ax.barh(range(len(words_plot)), vals, color=colors)
    ax.set_yticks(range(len(words_plot)))
    ax.set_yticklabels(words_plot, fontsize=10) 
    ax.invert_yaxis()
    ax.axvline(x=0, color='black', linewidth=1)
    ax.set_xlabel('Attribution Score', fontsize=11)
    ax.set_title(title, fontsize=12, fontweight='bold')
    
    legend_elements = [
        Patch(facecolor=GREEN, label='Pushes towards AI'),
        Patch(facecolor=RED, label='Pushes towards Human')
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def plot_top_tokens_by_type(top_tokens_per_type, target_label_map=None,
                             suptitle='So sánh Top Attribution Tokens theo loại'):
    if target_label_map is None:
        target_label_map = {'human': 0, 'rewrited': 1, 'generated': 1}

    n = len(top_tokens_per_type)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 6))
    if n == 1:
        axes = [axes]

    for ax, (gen_type, tokens) in zip(axes, top_tokens_per_type.items()):
        toks, vals = zip(*tokens)
        target = target_label_map.get(gen_type, 1)
        color = GREEN if target == 1 else RED
        direction = 'AI' if target == 1 else 'Human'

        ax.barh(range(len(toks)), vals, color=color)
        ax.set_yticks(range(len(toks)))
        ax.set_yticklabels(toks, fontsize=10)
        ax.invert_yaxis()
        ax.axvline(x=0, color='black', linewidth=0.5)
        ax.set_xlabel('Attribution Score')
        ax.set_title(f'{gen_type.upper()}\n(đẩy về phía "{direction}")', fontweight='bold')

    plt.suptitle(suptitle, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()
    plt.close(fig)

def highlight_text(words, scores, title="Attribution Highlight",
                    pred_label=None, pred_prob=None, true_label=None):
    if len(words) == 0:
        print("No words to highlight")
        return
 
    scores = np.array(scores)
    max_abs = np.max(np.abs(scores)) if np.max(np.abs(scores)) > 0 else 1.0
 
    spans = []
    for word, score in zip(words, scores):
        intensity = min(abs(score) / max_abs, 1.0)
 
        if score > 0:
            color = f"rgba(46, 204, 113, {0.15 + 0.65 * intensity:.2f})"
        else:
            color = f"rgba(231, 76, 60, {0.15 + 0.65 * intensity:.2f})"
 
        safe_word = html_lib.escape(word)
        spans.append(
            f'<span style="background-color:{color}; padding:2px 3px; '
            f'border-radius:3px; margin:1px;" title="score={score:+.3f}">{safe_word}</span>'
        )
 
    sentence_html = ' '.join(spans)
 
    header = f"<b>{html_lib.escape(title)}</b><br>"
    if pred_label is not None:
        header += f"Predicted: <b>{pred_label}</b>"
        if pred_prob is not None:
            header += f" (prob={pred_prob:.3f})"
        if true_label is not None:
            header += f" | True: <b>{true_label}</b>"
        header += "<br><br>"
 
    legend = (
        '<div style="margin-top:8px; font-size:12px;">'
        '<span style="background-color:rgba(46,204,113,0.7); padding:2px 6px;">xanh</span> = đẩy về phía AI &nbsp;&nbsp;'
        '<span style="background-color:rgba(231,76,60,0.7); padding:2px 6px;">đỏ</span> = đẩy về phía Human'
        '</div>'
    )
 
    full_html = f'<div style="font-family:sans-serif; font-size:15px; line-height:2.1;">{header}{sentence_html}{legend}</div>'
 
    display(HTML(full_html))