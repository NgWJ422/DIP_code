import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("../assets/castle.jpg", cv2.IMREAD_GRAYSCALE)

# Define the kernels K and K2
K = np.array([[1, 1, 1],
              [0, 0, 0],
              [-1, -1, -1]])

K2 = np.array([[1, 0, -1],
               [1, 0, -1],
               [1, 0, -1]])

# Apply both kernels to the image using filter2D
filtered_image_k = cv2.filter2D(image, -1, K)
filtered_image_k2 = cv2.filter2D(image, -1, K2)


#Combine pictues after filter, DO NOT ADD BOTH KERNAL TOGETHER
# Combine the two filtered images
combined_image = cv2.addWeighted(filtered_image_k, 0.5, filtered_image_k2, 0.5, 0)

# Display the original and combined images using Matplotlib
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(1, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Filtered Image (K) for horizontal edges
plt.subplot(1, 3, 2)
plt.imshow(filtered_image_k, cmap='gray')
plt.title('Filtered Image (K - Horizontal Edges)')
plt.axis('off')

# Combined Image
plt.subplot(1, 3, 3)
plt.imshow(combined_image, cmap='gray')
plt.title('Combined Edge Detection (K + K2)')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
