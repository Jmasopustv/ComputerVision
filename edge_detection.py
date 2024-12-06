import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load 
image_path = 'hart.jpg'
image = cv2.imread(image_path)

# greyscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# detect edges
edges = cv2.Canny(gray_image, threshold1=100, threshold2=200)

# Display
plt.figure(figsize=(14,7))

plt.subplot(121)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('Original Image')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Edge Detected Image')
plt.axis('off')

# close easily
def on_key(event):
    if event.key == '0':
        plt.close() 

# make key press talk to handler function
plt.gcf().canvas.mpl_connect('key_press_event', on_key)

plt.show()