import cv2

# Load the original color image
image = cv2.imread('cars-city-traffic-daylight_23-2149092089.jpg')

# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Get the dimensions of the original image
    height, width, channels = image.shape
    print(f"Original Image - Width: {width}, Height: {height}, Channels: {channels}")
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Get the dimensions of the grayscale image
    height_gray, width_gray = grayscale_image.shape  # Grayscale images have only 2 dimensions (no channels)
    print(f"Grayscale Image - Width: {width_gray}, Height: {height_gray}, Channels: 1")
    
    # Display the grayscale image in a window
    cv2.imshow('Grayscale Image', grayscale_image)

    # Wait for a key press to close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()
