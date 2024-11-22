import cv2

# Load the image
image = cv2.imread("../assets/dark_img.jpg")

# Check if the image is colored (3 channels) and convert it to grayscale
if image is not None and len(image.shape) == 3:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    # If already grayscale, just assign it to grayscale_image
    grayscale_image = image

# Apply histogram equalization to improve contrast
equalized_image = cv2.equalizeHist(grayscale_image)

# Display the grayscale and the equalized images
cv2.imshow('Grayscale Image', grayscale_image)
cv2.imshow('Histogram Equalized Image', equalized_image)

# Wait for a key press and close the image windows
cv2.waitKey(0)
cv2.destroyAllWindows()
