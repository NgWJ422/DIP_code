import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function for contrast stretching
def contrast_stretching(image):
    # Get min and max intensity values from the image
    min_val = np.min(image)
    max_val = np.max(image)

    # Apply contrast stretching formula
    stretched = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)
    return stretched

# Load grayscale image
image = cv2.imread("../assets/xray2.jpg", cv2.IMREAD_GRAYSCALE)
if image is None:
    raise ValueError("Image not found. Please check the path.")

# Apply contrast stretching
stretched_image = contrast_stretching(image)

# Apply histogram equalization
equalized_image = cv2.equalizeHist(image)

# Plot results
plt.figure(figsize=(15, 10))

# Original image and its histogram
plt.subplot(3, 2, 1)
plt.imshow(image, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(3, 2, 2)
plt.hist(image.ravel(), bins=256, range=(0, 256), color='black')
plt.title("Original Histogram")

# Contrast-stretched image and its histogram
plt.subplot(3, 2, 3)
plt.imshow(stretched_image, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(3, 2, 4)
plt.hist(stretched_image.ravel(), bins=256, range=(0, 256), color='blue')
plt.title("Contrast-Stretched Histogram")

# Histogram-equalized image and its histogram
plt.subplot(3, 2, 5)
plt.imshow(equalized_image, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(3, 2, 6)
plt.hist(equalized_image.ravel(), bins=256, range=(0, 256), color='green')
plt.title("Histogram-Equalized Histogram")


plt.tight_layout()
plt.show()
