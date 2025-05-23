import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def generate_noisy_images(original_path, k=20):
    """
    Generates k noisy versions of original image
    """
    original = np.array(Image.open(original_path).convert('L'))
    height, width = original.shape
    noisy_images = []
    
    for _ in range(k):
        # Generate Gaussian noise
        noise = np.random.normal(0, 1, (height, width))
        noisy = original + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        noisy_images.append(noisy)
    
    return noisy_images

def denoise_images(noisy_images, k_values):
    """
    Denoises by averaging different numbers of noisy images
    Returns denoised images and reconstruction errors
    """
    original = noisy_images[0]  # Assuming first image is original
    results = []
    errors = []
    
    for k in k_values:
        # Average first k images
        denoised = np.mean(noisy_images[:k], axis=0).astype(np.uint8)
        error = np.mean((original - denoised) ** 2)
        results.append(denoised)
        errors.append(error)
    
    return results, errors

noisy_images = generate_noisy_images('img/Redbull-3.jpg', k=20)
k_values = [1, 2, 5, 10, 15, 20]
denoised_images, errors = denoise_images(noisy_images, k_values)

plt.plot(k_values, errors)
plt.xlabel('Number of averaged images (k)')
plt.ylabel('Reconstruction Error (MSE)')
plt.title('Denoising Performance vs Number of Images')
plt.grid(True)
plt.show()