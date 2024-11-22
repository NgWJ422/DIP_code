import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Load the image (change 'your_image.jpg' to your image file path)
image_path = '../assets/castle.jpg'
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError("Image not found. Please check the file path.")

# Define kernels
kernels = {
    "Mean Filter": np.ones((3, 3), dtype=np.float32) / 9,
    "Gaussian Filter": (1/16) * np.array([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=np.float32),
    "High-Pass Filter": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32),
    "Unsharp Mask": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32),
    "Laplacian": np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=np.float32),
    "Sobel (Horizontal)": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32),
    "Sobel (Vertical)": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32),
    "Prewitt (Horizontal)": np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32),
    "Prewitt (Vertical)": np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32),
    "Roberts X": np.array([[1, 0], [0, -1]], dtype=np.float32),
    "Roberts Y": np.array([[0, 1], [-1, 0]], dtype=np.float32),
    "LoG (Laplacian of Gaussian)": np.array([
        [0, 0, -1, 0, 0],
        [0, -1, -2, -1, 0],
        [-1, -2, 16, -2, -1],
        [0, -1, -2, -1, 0],
        [0, 0, -1, 0, 0]
    ], dtype=np.float32),
}

# Create an output directory
output_dir = "filtered_images"
os.makedirs(output_dir, exist_ok=True)

# Function to apply filter and save image
def apply_filter_and_save(image, kernel, filter_name, output_dir):
    filtered_image = cv2.filter2D(image, -1, kernel)
    output_filename = os.path.join(output_dir, f"{filter_name.replace(' ', '_').lower()}.png")
    cv2.imwrite(output_filename, filtered_image)
    return filtered_image, output_filename

# Apply filters and display results
for filter_name, kernel in kernels.items():
    # Apply the filter
    filtered_image, output_filename = apply_filter_and_save(image, kernel, filter_name, output_dir)
    
    # Create a new Matplotlib figure for each filter
    plt.figure(figsize=(6, 6))
    plt.imshow(filtered_image, cmap='gray')
    plt.title(filter_name)
    plt.axis('off')
    
    # Show the image in a new window
    plt.show()
    print(f"Saved: {output_filename}")
