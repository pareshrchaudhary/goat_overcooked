import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

def print_env_stats(env_infos, episode, best_reward):
    latest_reward = env_infos['episode_rewards'][episode]
    print(f"Env Info:")
    print(f"Rewards: Max: {latest_reward['max']:.2f}, Mean ± Std: {latest_reward['mean']:.2f} ± {latest_reward['std']:.2f}, Min: {latest_reward['min']:.2f}, Best So Far: {best_reward:.2f}")
    if 'best_z' in env_infos:
        best_z = env_infos['best_z']
        print(f"Best Z: Max: {best_z['max']:.2f}, Mean ± Std: {best_z['mean']:.2f} ± {best_z['std']:.2f}, Min: {best_z['min']:.2f}")

def print_training_stats(train_info):
    print(f"\nTraining Info:")
    print(f"  Loss: {train_info['loss']:.3f}")
    print(f"  Learning Rate: {train_info['lr']}")
    print(f"  KL Divergence: {train_info['kl_divergence']:.3f}")
    print(f"  Entropy: {train_info['entropy']['value']:.3f}")

    # print(f"  Z Sample:\n"
    #         f"    Mean: {train_info['z_sample']['mean']:.4f}, Std: {train_info['z_sample']['std']:.4f}\n"
    #         f"    Max: {train_info['z_sample']['max']:.4f}, Min: {train_info['z_sample']['min']:.4f}\n")
    
    # print(f"  Advantages:\n"
    #       f"    Mean: {train_info['advantages']['mean']:.3f}, Std: {train_info['advantages']['std']:.3f}\n"
    #       f"    Max: {train_info['advantages']['max']:.3f}, Min: {train_info['advantages']['min']:.3f}\n")
    
    # print(f"  Log Probabilities:\n"
    #       f"    Mean: {train_info['log_prob']['mean']:.3f}, Std: {train_info['log_prob']['std']:.3f}\n"
    #       f"    Max: {train_info['log_prob']['max']:.3f}, Min: {train_info['log_prob']['min']:.3f}\n")

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

def z_distribution(z_sample_batch, print_z_stats=False, plot_z_hist=False):
    z_stats = {
                "mean": float(z_sample_batch.mean()),
                "std": float(z_sample_batch.std()),
                "min": float(z_sample_batch.min()),
                "max": float(z_sample_batch.max()),
                "ranges": {
                            "top": {
                                "34%": float(torch.quantile(z_sample_batch, 0.66)),
                                "13.6%": float(torch.quantile(z_sample_batch, 0.864)),
                                "2.1%": float(torch.quantile(z_sample_batch, 0.979)),
                                "0.1%": float(torch.quantile(z_sample_batch, 0.999))
                            },
                            "low": {
                                "34%": float(torch.quantile(z_sample_batch, 0.34)),
                                "13.6%": float(torch.quantile(z_sample_batch, 0.136)), 
                                "2.1%": float(torch.quantile(z_sample_batch, 0.021)),
                                "0.1%": float(torch.quantile(z_sample_batch, 0.001))
                            }
                        }
            }

    if print_z_stats:
        print(f"Z sample stats:")
        print(f"Mean ± Std: {z_stats['mean']:.4f} ± {z_stats['std']:.4f}")
        print(f"Min to Max: {z_stats['min']:.4f} to {z_stats['max']:.4f}")
        print(f"34% range: [{z_stats['ranges']['low']['34%']:.4f}, {z_stats['ranges']['top']['34%']:.4f}]")
        print(f"13.6% range: [{z_stats['ranges']['low']['13.6%']:.4f}, {z_stats['ranges']['top']['13.6%']:.4f}]")
        print(f"2.1% range: [{z_stats['ranges']['low']['2.1%']:.4f}, {z_stats['ranges']['top']['2.1%']:.4f}]")
        print(f"0.1% range: [{z_stats['ranges']['low']['0.1%']:.4f}, {z_stats['ranges']['top']['0.1%']:.4f}]")

    if plot_z_hist:
        z_vis_data = z_sample_batch.detach().cpu().numpy().flatten()
        fig = plt.figure(figsize=(8, 4))

        plt.axvspan(z_stats['ranges']['low']['34%'], z_stats['ranges']['top']['34%'], alpha=0.4, color='lightgreen', label='34% (±1σ)')
        plt.axvspan(z_stats['ranges']['low']['13.6%'], z_stats['ranges']['top']['13.6%'], alpha=0.3, color='lightblue', label='13.6% (±2σ)')
        plt.axvspan(z_stats['ranges']['low']['2.1%'], z_stats['ranges']['top']['2.1%'], alpha=0.2, color='peachpuff', label='2.1% (±3σ)')
        plt.axvspan(z_stats['ranges']['low']['0.1%'], z_stats['ranges']['top']['0.1%'], alpha=0.1, color='pink', label='0.1%')

        plt.hist(z_vis_data, bins=50, density=True, alpha=0.7, color='gray', edgecolor='black', weights=np.ones_like(z_vis_data)*100)
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.title(f'Distribution of Z Sample Batch Values\nMean: {z_stats["mean"]:.4f} ± {z_stats["std"]:.4f}\nMin: {z_stats["min"]:.4f}, Max: {z_stats["max"]:.4f}')
        plt.xlabel('Z Value')
        plt.ylabel('Percentage (%)')
        plt.grid(True)
        plt.legend()
        plt.show()

    return z_stats

def analyze_z_percentiles(episode_rewards_list, map_ea2z, print_z_percentile=False, plot_z_percentile=False):
    percentile_metrics = {
        'p34_low': [],  # -1σ: 15.9%
        'p34_high': [], # +1σ: 84.1%
        'p13_low': [],  # -2σ: 2.3%
        'p13_high': [], # +2σ: 97.7%
        'p02_low': [],  # -3σ: 0.1%
        'p02_high': []  # +3σ: 99.9%
    }

    z_reward_mapping = {
        'p34_low': [],
        'p34_high': [],
        'p13_low': [],
        'p13_high': [],
        'p02_low': [],
        'p02_high': []
    }

    for episode_reward in episode_rewards_list:
        percentile_metrics['p34_low'].append(np.percentile(episode_reward, 15.9))
        percentile_metrics['p34_high'].append(np.percentile(episode_reward, 84.1))
        percentile_metrics['p13_low'].append(np.percentile(episode_reward, 2.3))
        percentile_metrics['p13_high'].append(np.percentile(episode_reward, 97.7))
        percentile_metrics['p02_low'].append(np.percentile(episode_reward, 0.1))
        percentile_metrics['p02_high'].append(np.percentile(episode_reward, 99.9))
        
        curr_p34_low_idx = np.where(episode_reward <= np.percentile(episode_reward, 15.9))[0][0]
        curr_p34_high_idx = np.where(episode_reward <= np.percentile(episode_reward, 84.1))[0][0]
        curr_p13_low_idx = np.where(episode_reward <= np.percentile(episode_reward, 2.3))[0][0]
        curr_p13_high_idx = np.where(episode_reward <= np.percentile(episode_reward, 97.7))[0][0]
        curr_p02_low_idx = np.where(episode_reward <= np.percentile(episode_reward, 0.1))[0][0]
        curr_p02_high_idx = np.where(episode_reward <= np.percentile(episode_reward, 99.9))[0][0]

        z_reward_mapping['p34_low'].append(map_ea2z[(curr_p34_low_idx, 0)])
        z_reward_mapping['p34_high'].append(map_ea2z[(curr_p34_high_idx, 0)])
        z_reward_mapping['p13_low'].append(map_ea2z[(curr_p13_low_idx, 0)])
        z_reward_mapping['p13_high'].append(map_ea2z[(curr_p13_high_idx, 0)])
        z_reward_mapping['p02_low'].append(map_ea2z[(curr_p02_low_idx, 0)])
        z_reward_mapping['p02_high'].append(map_ea2z[(curr_p02_high_idx, 0)])

    
    if print_z_percentile:
        for percentile, z_values in z_reward_mapping.items():
            print(f"{percentile}:")
            print(f"  Mean: {np.mean(z_values):.4f}")
            print(f"  Std: {np.std(z_values):.4f}")
            print(f"  Min: {np.min(z_values):.4f}")
            print(f"  Max: {np.max(z_values):.4f}")

    if plot_z_percentile:
        fig, ax = plt.subplots(figsize=(8, 4))
        steps = range(1, len(percentile_metrics['p34_low']) + 1)

        for metric, values in percentile_metrics.items():
            ax.plot(steps, values, label=metric, linewidth=2.5)
        ax.set_xlabel('Step')
        ax.set_ylabel('Reward')
        ax.set_title('Std Statistics')
        ax.legend()
        ax.grid(True)

        plt.tight_layout()
        plt.show()

    return percentile_metrics, z_reward_mapping

