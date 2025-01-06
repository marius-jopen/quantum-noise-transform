import os
import torch
import numpy as np
from pathlib import Path

def transform_qn_samples_to_latent_for_audio():
    # Match paths with analyze script
    current_dir = Path(__file__).parent.parent.absolute()
    input_dir = current_dir / "quantum-noise/input/qn-full/audio-1"
    output_dir = current_dir / "quantum-noise/output/qn-full-latent/audio-1"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for file_path in input_dir.glob("*.csv"):
        try:
            print(f"\nProcessing file: {file_path}")
            
            # Load the CSV data
            data = np.loadtxt(file_path, delimiter=',')
            rows, cols = data.shape
            total_values = rows * cols
            
            print(f"CSV Data Analysis:")
            print(f"Rows: {rows:,}")
            print(f"Columns: {cols:,}")
            print(f"Total values: {total_values:,}")
            print(f"CSV file size: {file_path.stat().st_size / (1024*1024*1024):.2f} GB")
            
            # Convert to torch tensor and flatten
            tensor_data = torch.from_numpy(data.astype(np.float32)).flatten()
            print(f"Tensor values: {tensor_data.numel():,}")
            
            # Verify divisibility by 64
            if tensor_data.numel() % 64 != 0:
                print(f"WARNING: Total values {tensor_data.numel():,} is not divisible by 64!")
            
            # Reshape into one big noise tensor
            sequence_length = tensor_data.numel() // 64  # divide all values by number of channels
            noise_tensor = tensor_data.reshape(1, 64, sequence_length)
            
            print(f"Final tensor shape: {noise_tensor.shape}")
            
            # Save the transformed noise tensor
            output_filename = output_dir / f"{file_path.stem}-noise-full.pt"
            torch.save(noise_tensor, output_filename)
            
            print(f"Saved to: {output_filename}")
            print(f"File exists: {output_filename.exists()}")
            print(f"File size: {output_filename.stat().st_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting transformation process...")
    transform_qn_samples_to_latent_for_audio()
    print("Transformation process completed.")
