import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the image
image = cv2.imread("../assets/castle.jpg", cv2.IMREAD_GRAYSCALE)

# Apply Histogram Equalization
equalized_image = cv2.equalizeHist(image)

# Define the kernels K and K2
sobel_v = np.array([[-1, -2, -1],
              [0, 0, 0],
              [1, 2, 1]])

sobel_h = np.array([[-1, 0, 1],
               [-2, 0, 2],
               [-1, 0, 1]])

sobel_h_filtered = cv2.filter2D(equalized_image, -1, sobel_h)

sobel_v_filtered = cv2.filter2D(equalized_image, -1, sobel_v)

#Combine pictues after filter, DO NOT ADD BOTH KERNAL TOGETHER
combined_image = cv2.addWeighted(sobel_h_filtered, 0.5, sobel_v_filtered, 0.5, 0)



# Display the original and filtered images using Matplotlib
plt.figure(figsize=(15, 5))

# Original Image
plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Combined Filtered Image
plt.subplot(2, 3, 2)
plt.imshow(equalized_image, cmap='gray')
plt.title('Equalized Image')
plt.axis('off')

# Combined Filtered Image
plt.subplot(2, 3, 3)
plt.imshow(sobel_h_filtered, cmap='gray')
plt.title('sobel h')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.imshow(sobel_v_filtered, cmap='gray')
plt.title('sobel v')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(combined_image, cmap='gray')
plt.title('combined')
plt.axis('off')

# Show the plot
plt.tight_layout()
plt.show()
