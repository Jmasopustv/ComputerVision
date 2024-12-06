import cv2
import numpy as np

def detect_and_overlay_color(image_path):
    # Load
    image = cv2.imread(image_path)
    
    # Convert to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # color ranges (HSV)
    red_lower = np.array([0, 50, 50])
    red_upper = np.array([10, 255, 255])
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])
    green_lower = np.array([40, 50, 50])
    green_upper = np.array([90, 255, 255])
    
    # masks for each color
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)
    green_mask = cv2.inRange(hsv, green_lower, green_upper)
    
    # check area of each mask for color
    red_area = np.sum(red_mask)
    yellow_area = np.sum(yellow_mask)
    green_area = np.sum(green_mask)
    
    if max(red_area, yellow_area, green_area) == red_area:
        color = "Red"
    elif max(red_area, yellow_area, green_area) == yellow_area:
        color = "Yellow"
    elif max(red_area, yellow_area, green_area) == green_area:
        color = "Green"
    else:
        color = "Color not detected"
    
    # Overlay 
    font = cv2.FONT_HERSHEY_SIMPLEX
    position = (50, 50)
    font_scale = 1
    font_color = (255, 255, 255)  # White 
    line_type = 2
    cv2.putText(image, color, position, font, font_scale, font_color, line_type)
    
    # Display 
    cv2.imshow("Detection Result - " + color, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

image_paths = [
    'red_light.jpg',
    'yellow_light.jpg',
    'green_light.jpg'
]

# Process 
for path in image_paths:
    detect_and_overlay_color(path)
