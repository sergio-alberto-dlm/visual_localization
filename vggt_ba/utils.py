import random

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def open_json_retrieval(path):
    df = pd.read_json(path, lines=True)
    return df


def read_query_with_refs(sample):
    query_path = sample['query_path']
    query_topk = sample['topk']
    query_topk = {d['rank']: d for d in query_topk}

    vggt_input_ref_paths = [
        query_topk[i].get('abs_path')
        for i in range(1, 10)
    ]
    ranks = [
        query_topk[i].get('rank')
        for i in range(1, 10)
    ]
    score = [
        query_topk[i].get('score')
        for i in range(1, 10)
    ]

    vggt_input_paths =  [query_path] + vggt_input_ref_paths
    ranks = [0.0] + ranks
    score = [1.0] + score

    return vggt_input_paths, ranks, score


class JsonRetrieval:
    def __init__(self, path):
        self.df = open_json_retrieval(path)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> tuple:
        sample = self.df.iloc[idx]
        return read_query_with_refs(sample)

    def sample(self) -> tuple:
        idx = random.randint(0, len(self.df) - 1)
        return self[idx]


def plot_confidence_matrices_mpl(matrices):
    n, p, _ = matrices.shape
    n_cols = min(n, 5)
    n_rows = (n + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 5 * n_rows))
    axes = np.atleast_1d(axes).flatten()

    # Global min/max for shared color scale
    vmin = np.min(matrices)
    vmax = np.max(matrices)

    for i in range(n):
        mat = matrices[i]

        im = axes[i].imshow(mat, cmap='viridis', vmin=vmin, vmax=vmax)
        axes[i].set_title(f"Matrix {i+1}")
        axes[i].axis('off')

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    # Colorbar for consistent scale
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)

    plt.tight_layout(rect=[0, 0, 0.9, 1])
    plt.show()
