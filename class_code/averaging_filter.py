import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread('../assets/kid_in_gold.jpg')

# Convert the image from BGR to RGB (OpenCV loads images in BGR by default)
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Create the 5x5 averaging filter and apply it
kernel_5x5 = np.ones((5, 5), np.float32) / 25
image_5x5 = cv2.filter2D(image_rgb, -1, kernel_5x5)

# Create the 7x7 averaging filter and apply it
kernel_7x7 = np.ones((7, 7), np.float32) / 49
image_7x7 = cv2.filter2D(image_rgb, -1, kernel_7x7)

# Create the 15x15 averaging filter and apply it
kernel_15x15 = np.ones((15, 15), np.float32) / 225
image_15x15 = cv2.filter2D(image_rgb, -1, kernel_15x15)

# Plot the images using matplotlib
plt.figure(figsize=(10, 10))

# Display the original image
plt.subplot(2, 2, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.axis('off')

# Display the 5x5 averaged image
plt.subplot(2, 2, 2)
plt.imshow(image_5x5)
plt.title('5x5 Average Filter')
plt.axis('off')

# Display the 7x7 averaged image
plt.subplot(2, 2, 3)
plt.imshow(image_7x7)
plt.title('7x7 Average Filter')
plt.axis('off')

# Display the 15x15 averaged image
plt.subplot(2, 2, 4)
plt.imshow(image_15x15)
plt.title('15x15 Average Filter')
plt.axis('off')

# Show all images
plt.tight_layout()
plt.show()
