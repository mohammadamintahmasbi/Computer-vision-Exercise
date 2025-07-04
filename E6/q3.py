import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image in grayscale
image = cv2.imread('img/2.jpg', 0)  # 0 for grayscale
if image is None:
    raise FileNotFoundError("Image not found. Check the path!")

plt.imshow(image, cmap='gray'), plt.title('Original Image'), plt.show()

# Compute FFT and shift zero-frequency to the center
dft = np.fft.fft2(image)
dft_shift = np.fft.fftshift(dft)

# Visualize the magnitude spectrum (log scale for better visibility)
magnitude_spectrum = 20 * np.log(np.abs(dft_shift))
plt.imshow(magnitude_spectrum, cmap='gray'), plt.title('Magnitude Spectrum'), plt.show()

rows, cols = image.shape
mask = np.ones((rows, cols), np.uint8)  # Start with all 1s (keep all frequencies)

# Block horizontal lines (noise) in the FFT
center_row, center_col = rows // 2, cols // 2
mask[center_row - 2:center_row + 2, :] = 0  # Remove horizontal line (adjust thickness as needed)

# Visualize the mask
plt.imshow(mask, cmap='gray'), plt.title('Mask'), plt.show()

# Apply the mask
filtered_dft = dft_shift * mask

# Inverse FFT
idft_shift = np.fft.ifftshift(filtered_dft)
filtered_image = np.fft.ifft2(idft_shift).real

# Normalize and display
filtered_image = cv2.normalize(filtered_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
plt.imshow(filtered_image, cmap='gray'), plt.title('Filtered Image'), plt.show()