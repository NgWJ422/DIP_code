import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("../assets/castle.jpg")

# Convert the image to grayscale (since sharpening works on single channels)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Sharpening mask for 4 neighbouring pixels (vertical and horizontal neighbors)
mask_4_neighbors = np.array([[0, -1, 0],
                             [-1, 5, -1],
                             [0, -1, 0]])

# Sharpening mask for diagonal pixels
mask_diagonal = np.array([[-1, 0, -1],
                          [0, 5, 0],
                          [-1, 0, -1]])

# Sharpening mask for all 8 neighboring pixels
mask_all_neighbors = np.array([[-1, -1, -1],
                               [-1,  9, -1],
                               [-1, -1, -1]])

# Apply the filters to the grayscale image
sharpened_4_neighbors = cv2.filter2D(gray_image, -1, mask_4_neighbors)
sharpened_diagonal = cv2.filter2D(gray_image, -1, mask_diagonal)
sharpened_all_neighbors = cv2.filter2D(gray_image, -1, mask_all_neighbors)

# Create a figure to display the images
plt.figure(figsize=(12, 12))

# Original Image
plt.subplot(2, 2, 1)
plt.imshow(gray_image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Full 8 neighbor sharpening (All pixels)
plt.subplot(2, 2, 2)
plt.imshow(sharpened_all_neighbors, cmap='gray')
plt.title('Full 8 Neighbors Sharpened')
plt.axis('off')

# 4 Neighboring pixel sharpening (Vertical & Horizontal)
plt.subplot(2, 2, 3)
plt.imshow(sharpened_4_neighbors, cmap='gray')
plt.title('4 Neighbors Sharpened')
plt.axis('off')

# Diagonal pixel sharpening
plt.subplot(2, 2, 4)
plt.imshow(sharpened_diagonal, cmap='gray')
plt.title('Diagonal Neighbors Sharpened')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
