import cv2
import numpy as np

img = cv2.imread('img/2.png', cv2.IMREAD_COLOR)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray, 5)  # Reduce noise

# Detect circles (adjust parameters!)
circles = cv2.HoughCircles(
    gray,
    cv2.HOUGH_GRADIENT,
    dp=1,            # Resolution (1 = same as input)
    minDist=30,      # Minimum distance between circles
    param1=50,       # Edge detection threshold
    param2=30,       # Circle detection threshold (lower = more circles)
    minRadius=10,    # Smallest allowed radius
    maxRadius=100    # Largest allowed radius
)

# Draw detected circles
if circles is not None:
    circles = np.uint16(np.around(circles))
    output = img.copy()
    for (x, y, r) in circles[0, :]:
        cv2.circle(output, (x, y), r, (0, 255, 0), 2)
        cv2.circle(output, (x, y), 2, (0, 0, 255), 3)
        print(f"Circle: Center=({x}, {y}), Radius={r}")

cv2.imshow('Hough Circles', output)
cv2.waitKey(0)
cv2.destroyAllWindows()