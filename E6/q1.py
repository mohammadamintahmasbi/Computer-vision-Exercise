import cv2
import numpy as np

# 1. Load images (use correct paths!)
a = cv2.imread('img/a.jpg', 0)  # Grayscale
b = cv2.imread('img/b.jpg', 0)  # Grayscale

# Check if images loaded successfully
if a is None or b is None:
    raise FileNotFoundError("Image files not found. Check paths!")

# 2. Resize images to the same dimensions (e.g., smaller image's size)
height, width = min(a.shape[0], b.shape[0]), min(a.shape[1], b.shape[1])
a = cv2.resize(a, (width, height))
b = cv2.resize(b, (width, height))

# 3. Compute FFT
dft_a = np.fft.fft2(a)
dft_b = np.fft.fft2(b)

# 4. Extract magnitude and phase
mag_a, phase_a = np.abs(dft_a), np.angle(dft_a)
mag_b, phase_b = np.abs(dft_b), np.angle(dft_b)

# 5. Combine magnitude and phase
new1 = mag_a * np.exp(1j * phase_b)
new2 = mag_b * np.exp(1j * phase_a)

# 6. Inverse FFT
reconstructed1 = np.fft.ifft2(new1).real
reconstructed2 = np.fft.ifft2(new2).real

# 7. Normalize and display
reconstructed1 = cv2.normalize(reconstructed1, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
reconstructed2 = cv2.normalize(reconstructed2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

cv2.imshow('Reconstructed (Mag_A + Phase_B)', reconstructed1)
cv2.imshow('Reconstructed (Mag_B + Phase_A)', reconstructed2)
cv2.waitKey(0)
cv2.destroyAllWindows()