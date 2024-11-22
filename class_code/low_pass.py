import cv2
import numpy as np

# Load the image
#image = cv2.imread('assets/file_example_JPG_100kB.jpg')
image = cv2.imread('../assets/kid_in_gold.jpg')

# Check if the image is colored (3 channels)
if image is not None and len(image.shape) == 3:
    # Convert to grayscale if the image is colored
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    # If already grayscale, just assign it to grayscale_image
    grayscale_image = image

# Apply Gaussian filter with a kernel size of (5, 5) and standard deviation of 0
gaussian_filtered = cv2.GaussianBlur(grayscale_image, (5, 5), 0)

# Display the original and the filtered image
cv2.imshow('Original Image', image)
cv2.imshow('Gaussian Filtered Image', gaussian_filtered)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
