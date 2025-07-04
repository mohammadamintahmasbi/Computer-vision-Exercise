import cv2
import numpy as np

def gabor_highpass(image, ksize=31, sigma=5, theta=0, lambd=10, gamma=0.5):
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    kernel = kernel - kernel.mean()  # تبدیل به فیلتر بالاگذر
    filtered = cv2.filter2D(image, cv2.CV_8UC3, kernel)
    return filtered

image = cv2.imread('img/1.png', 0)
enhanced = gabor_highpass(image)
cv2.imshow('Enhanced', enhanced)
cv2.waitKey(0)