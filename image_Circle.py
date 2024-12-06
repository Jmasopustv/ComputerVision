import cv2
import numpy as np

# Load
image_path = 'balloons.jpg'
image = cv2.imread(image_path)
output = image.copy()
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gblur
gray_blurred = cv2.GaussianBlur(gray, (9, 9), 2)

# HT to detect
circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=30,
                            param1=100, param2=40, minRadius=10, maxRadius=100)


if circles is not None:
    # Convert coords and radius to integers
    circles = np.round(circles[0, :]).astype("int")

    # Loop
    for (x, y, r) in circles:
        # Draw on og
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        # Draw a rectangle on center point
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        
        # text
        cv2.putText(output, 'Circles detected: ' + str(len(circles)), (10, 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

# Display 
cv2.imshow("Detected Circles", output)
cv2.waitKey(0)
cv2.destroyAllWindows()

