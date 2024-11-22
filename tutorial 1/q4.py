import cv2

# Load the image
image = cv2.imread('lab-image.jpg')

# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Get the dimensions of the original image
    height, width, _ = image.shape

    # Display the original image in a window
    cv2.imshow('Original Image', image)

    # Calculate the center coordinates of the image
    center_x, center_y = width // 2, height // 2
    crop_size = 100
    x_start = center_x - crop_size // 2
    y_start = center_y - crop_size // 2

    # Crop a 100x100 pixel section from the center of the image
    cropped_image = image[y_start:y_start + crop_size, x_start:x_start + crop_size]

    # Display the cropped image
    cv2.imshow('Cropped Image (100x100)', cropped_image)

    # Resize the cropped image to 200x200 pixels
    resized_cropped_image = cv2.resize(cropped_image, (200, 200))

    # Display the resized cropped image
    cv2.imshow('Resized Cropped Image (200x200)', resized_cropped_image)

    # Wait for a key press and close all windows
    cv2.waitKey(0)
    cv2.destroyAllWindows()
