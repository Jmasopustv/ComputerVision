import cv2
import numpy as np
from matplotlib import pyplot as plt


def has_alternating_quadrants(x, y, radius, gray_img):
    
    radius = radius // 2
    # extract the circle's ROI with some padding
    pad = 10
    circle_img = gray_img[max(0, y-radius-pad):y+radius+pad, max(0, x-radius-pad):x+radius+pad]

    # check if the extracted ROI is valid
    if circle_img.shape[0] < radius*2 or circle_img.shape[1] < radius*2:
        return False

    # mean pixel value of the quadrants
    h, w = circle_img.shape
    cx, cy = w//2, h//2
    top_left = circle_img[cy-radius:cy, cx-radius:cx].mean()
    top_right = circle_img[cy-radius:cy, cx:cx+radius].mean()
    bottom_left = circle_img[cy:cy+radius, cx-radius:cx].mean()
    bottom_right = circle_img[cy:cy+radius, cx:cx+radius].mean()

    black_threshold, white_threshold = 100, 155
    # alternating black and white colors
    return ((top_left < black_threshold and bottom_right < black_threshold and
             top_right > white_threshold and bottom_left > white_threshold) or
            (top_left > white_threshold and bottom_right > white_threshold and
             top_right < black_threshold and bottom_left < black_threshold))


image_path = 'ps3-2-a_base.jpg' # works
# image_path = 'ps3-2-b_base.jpg' # partial br tr
# image_path = 'ps3-2-c_base.jpg' # partial tl
# image_path = 'ps3-2-d_base.jpg' # partial tr tl
# image_path = 'ps3-2-e_base.jpg' # none
# image_path = 'ps3-3-a_base.jpg' # none
# image_path = 'ps3-3-b_base.jpg' # partial br
# image_path = 'ps3-3-c_base.jpg' # partial tr

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=80,
                           param1=70, param2=30, minRadius=10, maxRadius=60)

# if circles are detected, verify each one
if circles is not None:
    circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles:
        print(f"Detected circle at (x={x}, y={y}, radius={r})")  
        if has_alternating_quadrants(x, y, r, gray):
            cv2.circle(image, (x, y), 10, (0, 0, 255), -1) # red dot
            cv2.circle(image, (x, y), r, (255, 0, 0), 3)  # blue circum.

# back to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.imshow(image_rgb)
plt.axis('off') 
plt.show()
