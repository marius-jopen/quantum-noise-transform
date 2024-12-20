import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import torch
import seaborn as sns
from scipy.stats import entropy

def analyze_noise_files():
    # Updated paths
    legacy_path = "quantum-noise/input/qn-latent-legacy"
    tester_latent_path = "quantum-noise/output/tester-latent"
    output_path = "quantum-noise/output/analytics-legacy-vs-tester"
    os.makedirs(output_path, exist_ok=True)

    # Process each legacy file
    for legacy_file in ['qn-latent-legacy-low.pt', 'qn-latent-legacy-medium.pt', 'qn-latent-legacy-high.pt']:
        base_name_no_ext = os.path.splitext(legacy_file)[0]
        print(f"\nProcessing legacy file: {legacy_file}")
        
        # Create figure with 5x2 grid (5 rows, 2 columns)
        fig = plt.figure(figsize=(15, 25))
        plt.suptitle(f"Analysis: {base_name_no_ext} vs Tester", fontsize=16)
        
        # Get files to compare
        files = {
            'Legacy Noise': (os.path.join(legacy_path, legacy_file), True),
            'Tester': (os.path.join(tester_latent_path, "tester-latent.pt"), True)
        }

        # First, collect all data to determine common scales
        all_data = {}
        all_normalized_data = {}
        
        for title, (file_path, is_torch) in files.items():
            if not os.path.exists(file_path):
                print(f"Warning: File not found: {file_path}")
                continue

            if is_torch:
                tensor = torch.load(file_path)
                if tensor.is_cuda:
                    tensor = tensor.cpu()
                data = tensor.numpy().flatten()
            else:
                data = np.loadtxt(file_path, delimiter=',').flatten()
            normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))
            
            all_data[title] = data
            all_normalized_data[title] = normalized_data

        # Find common scale for autocorrelation only
        max_autocorr = max([np.correlate(d[:1000], d[:1000], mode='full').max() for d in all_normalized_data.values()])

        # Plot with fixed/common scales
        for idx, (title, data) in enumerate(all_normalized_data.items()):
            # Row 1: Noise Pattern (1-2)
            ax = plt.subplot(5, 2, idx + 1)
            size = int(np.sqrt(len(data)))
            display_data = data[:size*size].reshape(size, size)
            ax.imshow(display_data, cmap='gray')
            ax.set_title(f"{title}\nNoise Pattern")
            ax.axis('off')

            # Row 2: Distribution (3-4)
            ax = plt.subplot(5, 2, idx + 3)
            sns.histplot(data, kde=True, ax=ax)
            ax.set_ylim(0, 2500)
            ax.set_title(f"{title}\nDistribution")

            # Row 3: Zoomed Distribution (5-6)
            ax = plt.subplot(5, 2, idx + 5)
            lower, upper = np.percentile(data, [5, 95])
            mask = (data >= lower) & (data <= upper)
            sns.histplot(data[mask], kde=True, ax=ax)
            ax.set_ylim(0, 200)
            ax.set_title(f"{title}\nZoomed Distribution")

            # Row 4: Autocorrelation (7-8)
            ax = plt.subplot(5, 2, idx + 7)
            autocorr = np.correlate(data[:1000], data[:1000], mode='full')
            ax.plot(autocorr[len(autocorr)//2:])
            ax.set_ylim(0, max_autocorr)
            ax.set_title(f"{title}\nAutocorrelation")

            # Row 5: Statistics (9-10)
            ax = plt.subplot(5, 2, idx + 9)
            stats_text = f"Statistics:\n"
            stats_text += f"Mean: {np.mean(all_data[title]):.4f}\n"
            stats_text += f"Std: {np.std(all_data[title]):.4f}\n"
            stats_text += f"Min: {np.min(all_data[title]):.4f}\n"
            stats_text += f"Max: {np.max(all_data[title]):.4f}\n"
            stats_text += f"Entropy: {entropy(data):.4f}"
            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, va='center')
            ax.axis('off')

        plt.tight_layout()
        # Save analysis
        output_file = os.path.join(output_path, f"{base_name_no_ext}_vs_tester_analysis.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    analyze_noise_files()
