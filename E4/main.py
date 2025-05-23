import cv2
import numpy as np
import matplotlib.pyplot as plt

def histogram_equalization(image):
    hist, bins = np.histogram(image.flatten(), 256, [0,256])
    
    cdf = hist.cumsum()
    cdf_normalized = cdf * hist.max() / cdf.max()
    
    cdf_m = np.ma.masked_equal(cdf,0)
    cdf_m = (cdf_m - cdf_m.min())*255/(cdf_m.max()-cdf_m.min())
    cdf = np.ma.filled(cdf_m,0).astype('uint8')
    
    equalized = cdf[image]
    
    return equalized

image = cv2.imread('img/p3286591407-5-800x533.jpg', 0)

equalized = histogram_equalization(image)

plt.figure(figsize=(10,5))
plt.subplot(121), plt.imshow(image, cmap='gray'), plt.title('input_image')
plt.subplot(122), plt.imshow(equalized, cmap='gray'), plt.title('moteadel shode')
plt.show()