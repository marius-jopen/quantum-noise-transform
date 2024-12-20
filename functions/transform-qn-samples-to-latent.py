import os
import torch
import numpy as np
from pathlib import Path

def transform_qn_samples_to_latent():
    # Define input and output directories
    input_dir = Path("quantum-noise/output/qn-samples")
    output_dir = Path("quantum-noise/output/qn-samples-latent")
    
    print(f"Looking for files in: {input_dir.absolute()}")
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each CSV file in the input directory
    for file_path in input_dir.glob("*.csv"):
        try:
            print(f"Processing file: {file_path}")
            
            # Load the CSV data
            data = np.loadtxt(file_path, delimiter=',')
            print(f"Loaded data shape: {data.shape}")
            
            # Convert to torch tensor
            tensor_data = torch.from_numpy(data.astype(np.float32))
            
            # Reshape directly to [1, 4, 128, 128]
            latent = tensor_data.reshape(1, 4, 128, 128)
            
            print(f"Transformed shape: {latent.shape}")
            
            # Create output filename (change extension from csv to pt)
            output_filename = output_dir / f"latent_{file_path.stem}.pt"
            
            # Save the transformed latent
            torch.save(latent, output_filename)
            
            print(f"Saved to: {output_filename}")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting transformation process...")
    transform_qn_samples_to_latent()
    print("Transformation process completed.")
