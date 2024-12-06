import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.spatial import distance as dist

image = cv2.imread('ps3-2-a_base.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

dst_focused = cv2.cornerHarris(gray, 9, 19, 0.07)
dst_focused_dilated = cv2.dilate(dst_focused, None)
focused_corners_threshold = dst_focused_dilated > 0.05 * dst_focused_dilated.max()

y_focused, x_focused = np.nonzero(focused_corners_threshold)
focused_corners = np.float32(list(zip(x_focused, y_focused)))

kmeans_focused = KMeans(n_clusters=4, random_state=42).fit(focused_corners)
focused_cluster_centers = kmeans_focused.cluster_centers_

def order_points(pts):

    rect = np.zeros((4, 2), dtype="float32")
    
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

ordered_points = order_points(focused_cluster_centers)

overlay_image = cv2.imread('kitten.jpeg') 
overlay_image = cv2.resize(overlay_image, (300, 300)) 

destination_points = np.array([[0, 0], [300, 0], [300, 300], [0, 300]], dtype=np.float32)

H, _ = cv2.findHomography(destination_points, ordered_points)

warped_overlay = cv2.warpPerspective(overlay_image, H, (image.shape[1], image.shape[0]))

mask = np.zeros_like(gray, dtype=np.uint8)
cv2.fillConvexPoly(mask, np.int32(ordered_points), 255)

background = cv2.bitwise_and(image, image, mask=~mask)

final_image = cv2.add(background, warped_overlay)

plt.figure(figsize=(10, 10))
plt.imshow(cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB))
plt.title('Final Image with Overlay')
plt.axis('off')
plt.show()
