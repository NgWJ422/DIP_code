import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load the grayscale image
image = cv2.imread('lizard.jpg', cv2.IMREAD_GRAYSCALE)

# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Original image (8-bit)
    original_image = image

    # Quantize to 4 bits (16 levels)
    quantized_4bit = (original_image // 16) * 16

    # Quantize to 2 bits (4 levels)
    quantized_2bit = (original_image // 64) * 64

    # Set up the figure and axes for Matplotlib
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Original image
    axes[0].imshow(original_image, cmap='gray', vmin=0, vmax=255)
    axes[0].set_title('Original 8-bit Image')
    axes[0].axis('off')  # Turn off axis

    # 4-bit Quantized image
    axes[1].imshow(quantized_4bit, cmap='gray', vmin=0, vmax=255)
    axes[1].set_title('Quantized 4-bit Image')
    axes[1].axis('off')

    # 2-bit Quantized image
    axes[2].imshow(quantized_2bit, cmap='gray', vmin=0, vmax=255)
    axes[2].set_title('Quantized 2-bit Image')
    axes[2].axis('off')

    # Adjust layout and show the images
    plt.tight_layout()
    plt.show()
