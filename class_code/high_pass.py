import cv2
import numpy as np

# Load the image
image = cv2.imread("../assets/laptop.jpg")

# Convert to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define the sharpening filter (kernel)
sharpening_kernel = np.array([[-1, -1, -1],
                              [-1,  9, -1],
                              [-1, -1, -1]])

# Apply the sharpening filter to the grayscale image
sharpened_gray_image = cv2.filter2D(gray_image, -1, sharpening_kernel)

# Display the grayscale and sharpened grayscale images
cv2.imshow("Grayscale Image", gray_image)
cv2.imshow("Sharpened Grayscale Image", sharpened_gray_image)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
