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
    qn_full_latent_path = "quantum-noise/output/qn-full-latent"
    output_path = "quantum-noise/output/analytics-qn-full-latent"
    os.makedirs(output_path, exist_ok=True)

    # Process each latent file
    for file_path in glob.glob(os.path.join(qn_full_latent_path, "*.pt")):
        base_name = os.path.basename(file_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        print(f"\nProcessing latent: {base_name}")
        
        # Create figure with 5x1 grid (only one column needed)
        fig = plt.figure(figsize=(8, 25))
        plt.suptitle(f"Analysis for latent: {base_name_no_ext}", fontsize=16)
        
        # Load and process the latent file
        tensor = torch.load(file_path)
        if tensor.is_cuda:
            tensor = tensor.cpu()
        data = tensor.numpy().flatten()
        normalized_data = (data - np.min(data)) / (np.max(data) - np.min(data))

        # Row 1: Noise Pattern
        ax = plt.subplot(5, 1, 1)
        size = int(np.sqrt(len(normalized_data)))
        display_data = normalized_data[:size*size].reshape(size, size)
        ax.imshow(display_data, cmap='gray')
        ax.set_title("Noise Pattern")
        ax.axis('off')

        # Row 2: Distribution
        ax = plt.subplot(5, 1, 2)
        sns.histplot(normalized_data, kde=True, ax=ax)
        ax.set_title("Distribution")

        # Row 3: Zoomed Distribution
        ax = plt.subplot(5, 1, 3)
        non_zero = normalized_data[normalized_data > 0.0001]
        if len(non_zero) > 0:
            sns.histplot(non_zero, kde=True, ax=ax)
            ax.set_ylim(0, 200)
        ax.set_title("Zoomed Distribution")

        # Row 4: Autocorrelation
        ax = plt.subplot(5, 1, 4)
        autocorr = np.correlate(normalized_data[:1000], normalized_data[:1000], mode='full')
        ax.plot(autocorr[len(autocorr)//2:])
        ax.set_title("Autocorrelation")

        # Row 5: Statistics
        ax = plt.subplot(5, 1, 5)
        stats_text = f"Statistics:\n"
        stats_text += f"Mean: {np.mean(data):.4f}\n"
        stats_text += f"Std: {np.std(data):.4f}\n"
        stats_text += f"Min: {np.min(data):.4f}\n"
        stats_text += f"Max: {np.max(data):.4f}\n"
        stats_text += f"Entropy: {entropy(normalized_data):.4f}"
        ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, va='center')
        ax.axis('off')

        plt.tight_layout()
        output_file = os.path.join(output_path, f"{base_name_no_ext}_analysis.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    analyze_noise_files()
