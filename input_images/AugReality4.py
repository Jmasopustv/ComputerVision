import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# image = cv2.imread('template_markers.png')
image = cv2.imread('ps3-2-a_base.jpg')
# image = cv2.imread('ps3-2-d_base.jpg')
# image = cv2.imread('ps3-2-e_base.jpg')
# image = cv2.imread('ps3-3-a_base.jpg')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# dst_focused = cv2.cornerHarris(gray, 2, 3, 0.04)
dst_focused = cv2.cornerHarris(gray, 9, 19, 0.07)
dst_focused_dilated = cv2.dilate(dst_focused, None)

focused_corners_threshold = dst_focused_dilated > 0.05 * dst_focused_dilated.max()  

y_focused, x_focused = np.nonzero(focused_corners_threshold)
focused_corners = np.float32(list(zip(x_focused, y_focused)))

kmeans_focused = KMeans(n_clusters=4, random_state=42).fit(focused_corners)
focused_cluster_centers = kmeans_focused.cluster_centers_

image_with_dots = image.copy()
for centroid in focused_cluster_centers:
    cv2.circle(image_with_dots, (int(centroid[0]), int(centroid[1])), 5, (0, 0, 255), -1)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(image_with_dots, cv2.COLOR_BGR2RGB))
plt.title('Overlay of Centroids as Red Dots')
plt.axis('off')
plt.show()


