import cv2
import numpy as np

# Load
image_path = 'rgb.jpg'
image = cv2.imread(image_path)

B, G, R = cv2.split(image)

black = np.zeros_like(B)

Blue = cv2.merge([B, black, black])
Green = cv2.merge([black, G, black])
Red = cv2.merge([black, black, R])

# Display
cv2.imshow('Blue Components', Blue)
cv2.imshow('Green Components', Green)
cv2.imshow('Red Components', Red)

# Close all windows easy
cv2.waitKey(0)
cv2.destroyAllWindows()

