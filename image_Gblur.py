import cv2

# Load 
image_path = 'rgb.jpg'
image = cv2.imread(image_path)

# Loop through 
for kernel_size in range(1, 20, 4):  # 1, 5, 9, 13, 17
    # Gblur
    blurred_image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
    
    # Display
    window_name = f'Blurred with kernel size {kernel_size}x{kernel_size}'
    cv2.imshow(window_name, blurred_image)
    
   
    cv2.waitKey(0)


cv2.destroyAllWindows()
