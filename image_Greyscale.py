import cv2

# Load
image_path = 'rgb.jpg'
image = cv2.imread(image_path)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Display
cv2.imshow('Grayscale Image', gray_image)

# Close all windows easy
cv2.waitKey(0)
cv2.destroyAllWindows()
