import cv2
import numpy as np
import matplotlib.pyplot as plt

def smooth_and_normalize(image, kernel_size=25, epsilon=1e-6):
    kernel = cv2.getGaussianKernel(kernel_size, -1)
    kernel = kernel * kernel.T
    
    smoothed = cv2.filter2D(image.astype(np.float32), -1, kernel)
    
    normalized = image.astype(np.float32) / (smoothed + epsilon)
    normalized = np.clip(normalized, 0, 255).astype(np.uint8)
    
    return smoothed, normalized


image = cv2.imread('img/p3286591407-5-800x533.jpg', 0)

smoothed, normalized = smooth_and_normalize(image)

plt.figure(figsize=(15,5))
plt.subplot(131), plt.imshow(image, cmap='gray'), plt.title('input_image')
plt.subplot(132), plt.imshow(smoothed, cmap='gray'), plt.title('make smooth')
plt.subplot(133), plt.imshow(normalized, cmap='gray'), plt.title('make normalize')
plt.show()