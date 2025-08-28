import cv2
import numpy as np

# خواندن تصویر
img = cv2.imread('img/1.png', 0)

# فیلتر پایین‌گذر گوسی
blur = cv2.GaussianBlur(img, (21, 21), 5)

# فیلتر بالاگذر = تصویر اصلی - تصویر بلور شده
highpass = cv2.subtract(img, blur)

# تقویت بالاگذر (High-boost)
A = 1.5  # ضریب تقویت
highboost = cv2.addWeighted(img, A, highpass, 1, 0)

# نمایش
cv2.imshow('Original', img)
cv2.imshow('High-pass', highpass)
cv2.imshow('High-boost', highboost)
cv2.waitKey(0)
cv2.destroyAllWindows()
