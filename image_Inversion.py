import cv2

# Load
image_path = 'street.jpg'
image = cv2.imread(image_path)

inverted_image = 255 - image

# Display
cv2.imshow('Original Image', image)
cv2.imshow('Inverted Image', inverted_image)

cv2.waitKey(0)
cv2.destroyAllWindows()
