import numpy as np
import glob
import os

# Set the number of samples to exactly match latent space requirements
SAMPLE_SIZE = 4 * 128 * 128  # = 65,536 (4 channels, 128x128 resolution)

# Create output directory if it doesn't exist
output_dir = "quantum-noise\output\qn-samples"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Process each file in the input directory
input_files = glob.glob("quantum-noise\input\qn-full/*.csv")

for file_path in input_files:
    print(f"\nProcessing {os.path.basename(file_path)}...")
    
    # Read the entire file content first
    with open(file_path, 'r') as f:
        # Replace newlines with commas and clean up any double commas
        content = f.read().replace('\n', ',').replace(',,', ',').strip(',')
        # Split by comma and convert to float array
        data = np.array([float(x.strip()) for x in content.split(',') if x.strip()])
    
    # Take first SAMPLE_SIZE values
    sampled_data = data[:SAMPLE_SIZE]
    
    # Save to new file in same format (comma-separated)
    base_name = os.path.basename(file_path).replace('qn-full-', '')  # Remove 'qn-full-' prefix
    output_path = os.path.join(output_dir, f"qn-sample-{base_name}")
    np.savetxt(output_path, [sampled_data], delimiter=',')  # Save as single row with commas
    
    print(f"Saved {len(sampled_data)} values to {os.path.basename(output_path)}")