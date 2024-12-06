import cv2
import numpy as np
from sklearn.cluster import KMeans

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

static_image_path = 'ps3-2-a_base.jpg'
static_image = cv2.imread(static_image_path)
gray_static_image = cv2.cvtColor(static_image, cv2.COLOR_BGR2GRAY)

dst_static = cv2.cornerHarris(gray_static_image, 9, 19, 0.07)
dst_static_dilated = cv2.dilate(dst_static, None)
static_corners_threshold = dst_static_dilated > 0.05 * dst_static_dilated.max()

y_static, x_static = np.nonzero(static_corners_threshold)
static_corners = np.float32(list(zip(x_static, y_static)))

kmeans_static = KMeans(n_clusters=4, random_state=42).fit(static_corners)
static_cluster_centers = kmeans_static.cluster_centers_

ordered_points = order_points(static_cluster_centers)

video_path = 'kitten.mp4' 
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (static_image.shape[1], static_image.shape[0]))
    
    h, w = frame.shape[:2]
    frame_points = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)

    H, _ = cv2.findHomography(frame_points, ordered_points)

    warped_frame = cv2.warpPerspective(frame, H, (static_image.shape[1], static_image.shape[0]))

    mask = np.zeros_like(gray_static_image, dtype=np.uint8)
    cv2.fillConvexPoly(mask, np.int32(ordered_points), 255)
    inv_mask = cv2.bitwise_not(mask)

    static_bg = cv2.bitwise_and(static_image, static_image, mask=inv_mask)

    combined_image = cv2.add(static_bg, warped_frame)

    cv2.imshow('Overlayed Video on Image', combined_image)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
