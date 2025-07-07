import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess
img = cv2.imread('img/1.png', 0)
_, bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Morphological cleaning
k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
opened = cv2.morphologyEx(bin, cv2.MORPH_OPEN, k, iterations=2)

# Custom kernel based on expected tooth shape
# Adjust this based on your actual gear teeth!
tooth_width = 5  # pixels - adjust based on your image
tooth_height = 10 # pixels - adjust based on your image
teeth_kernel = np.zeros((tooth_height, tooth_width), dtype=np.uint8)
teeth_kernel[0,:] = 1  # Top edge
teeth_kernel[-1,:] = 1 # Bottom edge
teeth_kernel[:,0] = 1  # Left edge
teeth_kernel[:,-1] = 1 # Right edge

hitmiss = cv2.morphologyEx(opened, cv2.MORPH_HITMISS, teeth_kernel)

# Find and filter contours
contours, _ = cv2.findContours(hitmiss, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
min_tooth_area = 20  # Adjust based on your image scale
filtered_contours = [c for c in contours if cv2.contourArea(c) > min_tooth_area]

# Visualization
result = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
cv2.drawContours(result, filtered_contours, -1, (0,255,0), 1)

plt.figure(figsize=(15,5))
plt.subplot(131), plt.imshow(bin, cmap='gray'), plt.title('Binary')
plt.subplot(132), plt.imshow(hitmiss, cmap='gray'), plt.title('Teeth Detection')
plt.subplot(133), plt.imshow(result), plt.title('Final Result')
plt.show()