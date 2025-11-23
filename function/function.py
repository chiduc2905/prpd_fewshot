import torch
import torch.nn as nn
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

def seed_func(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ContrastiveLoss(nn.Module):
    def __init__(self):
        super(ContrastiveLoss, self).__init__()
    
    def forward(self, output, target):
        # output: (batch_size, way) or (way,)
        # target: (batch_size,) or scalar
        if output.dim() == 1:
            upper = torch.exp(output[target])
            lower = torch.exp(output).sum()
            loss = -torch.log(upper / lower)
        else:
            # Gather scores for target classes
            # output[i, target[i]]
            upper = torch.exp(output.gather(1, target.view(-1, 1)).squeeze())
            lower = torch.exp(output).sum(dim=1)
            loss = -torch.log(upper / lower)
            loss = loss.mean()
        return loss

def plot_confusion_matrix(true_labels, predictions, num_classes=3, save_path=None):
    """
    Plot confusion matrix with style matching the reference image.
    Designed for 3-class classification.
    """
    conf_matrix = confusion_matrix(true_labels, predictions)
    
    # Calculate percentages
    row_sums = conf_matrix.sum(axis=1)[:, np.newaxis]
    conf_matrix_percent = conf_matrix.astype('float') / (row_sums + 1e-6) * 100
    
    # Create figure
    fig, ax = plt.subplots(figsize=(8, 7))
    
    # Create annotations with count and percentage
    annotations = np.empty_like(conf_matrix, dtype=object)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            count = conf_matrix[i, j]
            percent = conf_matrix_percent[i, j]
            annotations[i, j] = f'{count}\n({percent:.1f}%)'
    
    # Plot heatmap with green colormap
    sns.heatmap(conf_matrix, annot=annotations, fmt='', cmap='Greens', 
                cbar_kws={'label': ''}, 
                linewidths=2, linecolor='white', ax=ax,
                annot_kws={'size': 11, 'weight': 'bold'},
                vmin=0, square=True)
    
    # Customize labels and title
    ax.set_xlabel('Predicted Labels', fontsize=12, fontweight='normal')
    ax.set_ylabel('True Labels', fontsize=12, fontweight='normal')
    ax.set_title('Confusion Matrix', fontsize=13, fontweight='normal', pad=15)
    
    # Set tick labels
    ax.set_xticklabels(range(num_classes), fontsize=10)
    ax.set_yticklabels(range(num_classes), fontsize=10, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f'Confusion matrix saved to: {save_path}')
    plt.close()

def plot_tsne(features, labels, num_classes=3, save_path=None, use_pca=True, n_pca_components=50, class_names=None):
    """
    Plot t-SNE visualization.
    """
    print('Computing t-SNE...')
    
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.linewidth'] = 1.5
    
    if class_names is None:
        class_names = ['Corona', 'No PD', 'Surface']
    
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    features_normalized = scaler.fit_transform(features)
    
    if use_pca and features_normalized.shape[1] > n_pca_components:
        from sklearn.decomposition import PCA
        print(f'Applying PCA: {features_normalized.shape[1]} -> {n_pca_components} dimensions')
        pca = PCA(n_components=n_pca_components, random_state=42)
        features_normalized = pca.fit_transform(features_normalized)
    
    n_samples = features_normalized.shape[0]
    perplexity = min(50, max(5, n_samples // 10))
    
    try:
        from tsnecuda import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, verbose=0)
        features_tsne = tsne.fit_transform(features_normalized)
    except ImportError:
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, verbose=1, random_state=42)
        features_tsne = tsne.fit_transform(features_normalized)
    except Exception as e:
        print(f"Error with tsnecuda, using sklearn: {e}")
        from sklearn.manifold import TSNE
        tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, verbose=1, random_state=42)
        features_tsne = tsne.fit_transform(features_normalized)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    colors = sns.color_palette("husl", num_classes) if num_classes > 6 else ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    markers = ['o', 's', '^', 'D', 'v', '<']
    
    for label in range(num_classes):
        mask = labels == label
        if np.sum(mask) > 0:
            ax.scatter(features_tsne[mask, 0], features_tsne[mask, 1], 
                      c=[colors[int(label) % len(colors)]], 
                      marker=markers[int(label) % len(markers)],
                      label=class_names[int(label)] if int(label) < len(class_names) else f'Class {int(label)}', 
                      s=60, alpha=0.8, edgecolors='k', linewidth=0.5)
    
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('t-SNE Dimension 1', fontsize=14, fontweight='bold')
    ax.set_ylabel('t-SNE Dimension 2', fontsize=14, fontweight='bold')
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    legend = ax.legend(loc='best', frameon=True, fontsize=12, framealpha=0.9, edgecolor='black')
    legend.get_frame().set_linewidth(1.0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=600, bbox_inches='tight', facecolor='white')
        print(f't-SNE plot saved to: {save_path}')
    plt.close()
