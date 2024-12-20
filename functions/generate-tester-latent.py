import torch
import os

def manual_seed(seed):
    """Set the random seed for reproducibility"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def randn(seed, shape, generator=None):
    """USED IN FIRST STEP: Generates initial noise with a specific seed"""
    print(f"[NOISE -> randn] randn - seed={seed}, shape={shape}")
    
    manual_seed(seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[NOISE] Using PyTorch generator on {device}")
    return torch.randn(shape, device=device, generator=generator)

def generate_and_save_noise(seed, width=256, height=256, channels=1, output_dir="quantum-noise/output/tester-latent"):
    """Generates noise and saves it as a tensor
    Default size is 256x256 with 1 channel to get exactly 65,536 values"""
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate noise using the existing randn function
    shape = (1, channels, height, width)  # This will create 256x256=65,536 values
    noise = randn(seed, shape)
    
    # Save the raw tensor
    base_name = f"tester-latent"
    torch.save(noise, os.path.join(output_dir, f"{base_name}.pt"))
    
    print(f"Noise tensor saved in {output_dir}")
    return noise

if __name__ == "__main__":
    # Example usage
    seed = 42
    noise = generate_and_save_noise(seed)  # Will generate 65,536 values
