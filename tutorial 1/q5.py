import cv2
import matplotlib.pyplot as plt
import os

# Create a folder to save the images
output_folder = 'saved_images'
os.makedirs(output_folder, exist_ok=True)

# Load the high-resolution image
image = cv2.imread('forest-8371211.jpg')

# Check if the image was loaded properly
if image is None:
    print("Error: Image not found or unable to load.")
else:
    # Convert the image from BGR to RGB format
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Get and print the resolution of the original image
    original_height, original_width, _ = image.shape
    print(f"Original Image Resolution: {original_width} x {original_height}")

    # Downsample the image to 50% of its original resolution
    downsampled_50 = cv2.resize(image, (original_width // 2, original_height // 2))
    downsampled_50_rgb = cv2.cvtColor(downsampled_50, cv2.COLOR_BGR2RGB)
    height_50, width_50, _ = downsampled_50.shape
    print(f"50% Downsampled Image Resolution: {width_50} x {height_50}")

    # Downsample the image to 25% of its original resolution
    downsampled_25 = cv2.resize(image, (original_width // 4, original_height // 4))
    downsampled_25_rgb = cv2.cvtColor(downsampled_25, cv2.COLOR_BGR2RGB)
    height_25, width_25, _ = downsampled_25.shape
    print(f"25% Downsampled Image Resolution: {width_25} x {height_25}")

    # Downsample the image to 10% of its original resolution
    downsampled_10 = cv2.resize(image, (original_width // 10, original_height // 10))
    downsampled_10_rgb = cv2.cvtColor(downsampled_10, cv2.COLOR_BGR2RGB)
    height_10, width_10, _ = downsampled_10.shape
    print(f"10% Downsampled Image Resolution: {width_10} x {height_10}")

    # Save images to the output folder
    cv2.imwrite(os.path.join(output_folder, 'q5-original.jpg'),image)
    cv2.imwrite(os.path.join(output_folder, 'q5-50percent.jpg'),downsampled_50)
    cv2.imwrite(os.path.join(output_folder, 'q5-25percent.jpg'),downsampled_25)
    cv2.imwrite(os.path.join(output_folder, 'q5-10percent.jpg'),downsampled_10)

    # Function to display an image with zoom
    def display_image(image, title):
        plt.imshow(image)
        plt.title(title)
        plt.axis('on')  # Turn on axis to allow zooming
        plt.show()

    # Show each image separately
    display_image(image_rgb, 'Original Image')
    display_image(downsampled_50_rgb, '50% Downsampled Image')
    display_image(downsampled_25_rgb, '25% Downsampled Image')
    display_image(downsampled_10_rgb, '10% Downsampled Image')

print("Images saved in folder:", output_folder)
