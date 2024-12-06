import cv2
import numpy as np
import matplotlib.pyplot as plt

# image_path = 'template_markers.png'
image_path = 'ps3-2-a_base.jpg'
# image_path = 'ps3-2-d_base.jpg'

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

gray = np.float32(gray)
# dst = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
dst = cv2.cornerHarris(gray, blockSize=9, ksize=19, k=0.07)

dst = cv2.dilate(dst, None)

thresh = 0.01 * dst.max()
corner_image = np.copy(image)
corner_image[dst > thresh] = [0, 0, 255]

plt.figure(figsize=(10,10))
plt.imshow(cv2.cvtColor(corner_image, cv2.COLOR_BGR2RGB))
plt.show()
