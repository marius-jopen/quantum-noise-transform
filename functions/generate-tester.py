import os
import numpy as np

def generate_and_save_noise(seed, width=256, height=256, channels=1, output_dir="quantum-noise/output/tester"):
    """Generates noise and saves it as a CSV file
    Default size is 256x256 with 1 channel to get exactly 65,536 values"""
    
    # Set the random seed for reproducibility
    np.random.seed(seed)
    
    # Generate noise directly using numpy
    total_values = width * height * channels
    noise = np.random.normal(0, 1, total_values)
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Save as CSV on a single line
    base_name = f"tester"
    csv_path = os.path.join(output_dir, f"{base_name}.csv")
    
    with open(csv_path, 'w') as f:
        # Convert numbers to strings with scientific notation
        numbers_str = [f"{x:.18e}" for x in noise]
        # Join with commas and write
        f.write(','.join(numbers_str))
    
    print(f"Noise CSV saved in {csv_path}")
    return noise

if __name__ == "__main__":
    # Example usage
    seed = 42
    noise = generate_and_save_noise(seed)  # Will generate 65,536 values
