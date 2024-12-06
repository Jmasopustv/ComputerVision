import cv2
import numpy as np

# Load
image_path = 'street.jpg'
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gblur to make it less busy
blurred = cv2.GaussianBlur(gray, (7, 7), 0)

# Edge detection 
edges = cv2.Canny(blurred, 50, 150, apertureSize=3)

# Hough Line Transform
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

# Draw lines on og
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# Display
cv2.imshow('Image with Lines', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
