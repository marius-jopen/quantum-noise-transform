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
    qn_samples_path = "quantum-noise/output/qn-samples"
    qn_samples_latent_path = "quantum-noise/output/qn-samples-latent"
    tester_path = "quantum-noise/output/tester"
    tester_latent_path = "quantum-noise/output/tester-latent"
    output_path = "quantum-noise/output/analytics-qn-samples-and-tester"
    os.makedirs(output_path, exist_ok=True)

    # Process each sample file
    for base_file_path in glob.glob(os.path.join(qn_samples_path, "*.csv")):
        base_name = os.path.basename(base_file_path)
        base_name_no_ext = os.path.splitext(base_name)[0]
        print(f"\nProcessing sample: {base_name}")
        
        # Create figure with 5x4 grid (added one row)
        fig = plt.figure(figsize=(20, 25))  # Increased height for new row
        plt.suptitle(f"Analysis for sample: {base_name_no_ext}", fontsize=16)
        
        # Get corresponding files for this sample
        files = {
            'Quantum Noise': (base_file_path, False),
            'Quantum Noise Latent': (os.path.join(qn_samples_latent_path, f"latent_{base_name_no_ext}.pt"), True),
            'Tester': (os.path.join(tester_path, "tester.csv"), False),
            'Tester Latent': (os.path.join(tester_latent_path, "tester-latent.pt"), True)
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
            # Row 1: Noise Pattern (1-4)
            ax = plt.subplot(5, 4, idx + 1)
            size = int(np.sqrt(len(data)))
            display_data = data[:size*size].reshape(size, size)
            ax.imshow(display_data, cmap='gray')
            ax.set_title(f"{title}\nNoise Pattern")
            ax.axis('off')

            # Row 2: Distribution (5-8) with fixed scale
            ax = plt.subplot(5, 4, idx + 5)
            sns.histplot(data, kde=True, ax=ax)
            ax.set_ylim(0, 2500)
            ax.set_title(f"{title}\nDistribution")

            # Row 3: Zoomed Distribution (9-12) with fixed scale 200
            ax = plt.subplot(5, 4, idx + 9)
            if 'Quantum' in title:
                non_zero = data[data > 0.0001]
                if len(non_zero) > 0:
                    sns.histplot(non_zero, kde=True, ax=ax)
            else:
                lower, upper = np.percentile(data, [5, 95])
                mask = (data >= lower) & (data <= upper)
                sns.histplot(data[mask], kde=True, ax=ax)
            ax.set_ylim(0, 200)
            ax.set_title(f"{title}\nZoomed Distribution")

            # Row 4: Autocorrelation (13-16)
            ax = plt.subplot(5, 4, idx + 13)
            autocorr = np.correlate(data[:1000], data[:1000], mode='full')
            ax.plot(autocorr[len(autocorr)//2:])
            ax.set_ylim(0, max_autocorr)
            ax.set_title(f"{title}\nAutocorrelation")

            # Row 5: Statistics (17-20)
            ax = plt.subplot(5, 4, idx + 17)
            stats_text = f"Statistics:\n"
            stats_text += f"Mean: {np.mean(all_data[title]):.4f}\n"
            stats_text += f"Std: {np.std(all_data[title]):.4f}\n"
            stats_text += f"Min: {np.min(all_data[title]):.4f}\n"
            stats_text += f"Max: {np.max(all_data[title]):.4f}\n"
            stats_text += f"Entropy: {entropy(data):.4f}"
            ax.text(0.1, 0.5, stats_text, transform=ax.transAxes, va='center')
            ax.axis('off')

        plt.tight_layout()
        # Save this sample's analysis as one PNG
        output_file = os.path.join(output_path, f"{base_name_no_ext}_analysis.png")
        plt.savefig(output_file, bbox_inches='tight', dpi=300)
        plt.close()

if __name__ == "__main__":
    analyze_noise_files()
