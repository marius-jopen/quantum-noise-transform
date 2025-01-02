import os
import torch
import numpy as np
from pathlib import Path

def transform_qn_samples_to_latent():
    # Get the base directory (quantum-noise-transform)
    current_dir = Path(__file__).parent.parent.absolute()  # Go up from 'functions' to base
    
    # Set exact paths
    input_dir = current_dir / "input/qn-full"
    output_dir = current_dir / "output/qn-full-latent/v-2"
    
    print(f"Base directory: {current_dir}")
    print(f"Looking for input files in: {input_dir}")
    print(f"Will save output files to: {output_dir}")
    
    # Check if input directory exists
    if not input_dir.exists():
        print(f"ERROR: Input directory does not exist: {input_dir}")
        print("Creating input directory structure...")
        input_dir.mkdir(parents=True, exist_ok=True)
        return
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # List all files in input directory
    input_files = list(input_dir.glob("*.csv"))
    if not input_files:
        print(f"No CSV files found in {input_dir}")
        return
        
    print(f"Found {len(input_files)} CSV files: {[f.name for f in input_files]}")
    
    # Process each CSV file in the input directory
    for file_path in input_files:
        try:
            print(f"\nProcessing file: {file_path}")
            
            # Load the CSV data
            data = np.loadtxt(file_path, delimiter=',')
            print(f"Loaded data shape: {data.shape}")
            
            # Convert to torch tensor and flatten
            tensor_data = torch.from_numpy(data.astype(np.float32)).flatten()
            
            # Calculate size for a single latent
            single_latent_size = 4 * 128 * 128
            
            # Calculate how many complete latents we can create
            num_latents = tensor_data.numel() // single_latent_size
            print(f"Can create {num_latents} latent tensors")
            
            # Take maximum 100 latents to keep file size manageable
            num_latents = min(num_latents, 100)
            
            # Reshape into multiple latents
            latents = tensor_data[:num_latents * single_latent_size].reshape(num_latents, 4, 128, 128)
            
            print(f"Transformed shape: {latents.shape}")
            
            # Create output filename
            output_filename = output_dir / f"{file_path.stem}-latent.pt"
            
            # Save the transformed latents
            torch.save(latents, output_filename)
            
            print(f"Saved to: {output_filename}")
            print(f"File exists: {output_filename.exists()}")
            print(f"File size: {output_filename.stat().st_size / (1024*1024):.2f} MB")
            
        except Exception as e:
            print(f"Error processing {file_path.name}: {str(e)}")

if __name__ == "__main__":
    print("Starting transformation process...")
    transform_qn_samples_to_latent()
    print("Transformation process completed.")
