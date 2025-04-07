import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def generate_noisy_images(original_path, k=20, noise_variance=1):
    """
    Generates k noisy versions of original image with Gaussian noise
    Args:
        original_path: path to original image
        k: number of noisy images to generate
        noise_variance: variance of Gaussian noise
    Returns:
        original image array and list of noisy images
    """
    original = np.array(Image.open(original_path).convert('L'))
    height, width = original.shape
    noisy_images = []
    
    for _ in range(k):
        # Generate Gaussian noise (mean=0, specified variance)
        noise = np.random.normal(0, noise_variance, (height, width))
        noisy = original + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        noisy_images.append(noisy)
    
    return original, noisy_images

def denoise_and_visualize(original, noisy_images, k_values, save_dir='denoising_results'):
    """
    Denoises images with different k values and visualizes results
    Args:
        original: original image array
        noisy_images: list of noisy images
        k_values: list of k values to test
        save_dir: directory to save results
    """
    # Create output directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Prepare figure for visualization
    plt.figure(figsize=(15, 10))
    num_plots = len(k_values) + 1  # +1 for original
    
    # Display original image
    plt.subplot(2, (num_plots+1)//2, 1)
    plt.imshow(original, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    # Process each k value
    psnrs = []
    mses = []
    
    for i, k in enumerate(k_values, start=1):
        # Average first k noisy images
        denoised = np.mean(noisy_images[:k], axis=0).astype(np.uint8)
        
        # Calculate metrics
        mse = np.mean((original - denoised) ** 2)
        psnr = 20 * np.log10(255.0 / np.sqrt(mse)) if mse > 0 else float('inf')
        psnrs.append(psnr)
        mses.append(mse)
        
        # Save denoised image
        Image.fromarray(denoised).save(f'{save_dir}/denoised_k{k}.png')
        
        # Display in subplot
        plt.subplot(2, (num_plots+1)//2, i+1)
        plt.imshow(denoised, cmap='gray')
        plt.title(f'k={k}\nPSNR={psnr:.2f} dB\nMSE={mse:.2f}')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/denoising_comparison.png')
    plt.show()
    
    # Plot error metrics
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_values, mses, 'o-')
    plt.xlabel('Number of images (k)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(k_values, psnrs, 'o-')
    plt.xlabel('Number of images (k)')
    plt.ylabel('PSNR (dB)')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/error_metrics.png')
    plt.show()

original_img_path = 'img/Redbull-3.jpg'  
k_values = [1, 2, 5, 10, 15, 20]  # Different k values to test

# Generate noisy images
original, noisy_images = generate_noisy_images(original_img_path, k=max(k_values))

if noisy_images is not None:
    # Save noisy images (optional)
    for i, image in enumerate(noisy_images):
        Image.fromarray(image).save(f'noisy_image_{i}.png')

denoise_and_visualize(original, noisy_images, k_values)