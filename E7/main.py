import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load and preprocess
img = cv2.imread('img/1.png', 0)
_, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# Determine kernel size based on teeth size (adjust as needed)
kernel_size = 15  # Should be larger than tooth dimensions
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
# Find all contours
contours, _ = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Filter contours by size to find main gear body
main_body = max(contours, key=cv2.contourArea)

# Create mask of just the main body
mask = np.zeros_like(img)
cv2.drawContours(mask, [main_body], -1, 255, -1)

# Smooth edges if needed
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

cv2.imshow('Original', img)
cv2.imshow('Gear without Teeth', mask)
cv2.waitKey(0)