import os

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE


def tsne_plot_finite_groups(z: np.ndarray, names: list[str], plot_name: str, n_samples=-1, perp=10) -> None:
    if n_samples != -1:
        assert n_samples < len(z)
        z_mask = np.random.choice(len(z), n_samples, replace=False)
        z = z[z_mask]
        names = [names[i] for i in z_mask]
    z_2d = TSNE(n_components=2, perplexity=perp).fit_transform(z)
    plt.clf()

    for name in set(names):
        masks = np.array([name_i == name for name_i in names])
        if np.any(masks):
            plt.scatter(z_2d[masks, 0], z_2d[masks, 1], label=name)
    
    f_dir = os.path.join(os.path.dirname(__file__), "plots")
    os.makedirs(f_dir, exist_ok=True)
    plt.legend()
    plt.savefig(os.path.join(f_dir, plot_name))