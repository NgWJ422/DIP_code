import cv2
import numpy as np
import matplotlib.pyplot as plt

# Function to add Gaussian noise
def add_gaussian_noise(image, mean=0, std=25):
    gaussian = np.random.normal(mean, std, image.shape).astype('float32')
    noisy_image = image.astype('float32') + gaussian
    return np.clip(noisy_image, 0, 255).astype('uint8')

# Function to add Salt-and-Pepper noise
def add_salt_and_pepper_noise(image, salt_prob=0.01, pepper_prob=0.01):
    noisy_image = image.copy()
    total_pixels = image.size
    num_salt = int(salt_prob * total_pixels)
    num_pepper = int(pepper_prob * total_pixels)
    
    # Add salt
    salt_coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coords[0], salt_coords[1]] = 255
    
    # Add pepper
    pepper_coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coords[0], pepper_coords[1]] = 0

    return noisy_image

# Function to add Poisson noise
def add_poisson_noise(image):
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy_image = np.random.poisson(image.astype('float32') * vals) / vals
    return np.clip(noisy_image, 0, 255).astype('uint8')

# Function to add Speckle noise
def add_speckle_noise(image, std=0.1):
    speckle = np.random.randn(*image.shape) * std
    noisy_image = image.astype('float32') + image.astype('float32') * speckle
    return np.clip(noisy_image, 0, 255).astype('uint8')

# Load the image (grayscale)
image = cv2.imread('../assets/laptop.jpg', cv2.IMREAD_GRAYSCALE)

# Add different types of noise
noisy_images = {
    "Original": image,
    "Gaussian Noise": add_gaussian_noise(image),
    "Salt-and-Pepper Noise": add_salt_and_pepper_noise(image),
    "Poisson Noise": add_poisson_noise(image),
    "Speckle Noise": add_speckle_noise(image),
}

# Apply median filter
filtered_images = {name: cv2.medianBlur(noisy_image, 3) for name, noisy_image in noisy_images.items()}

# Display results using Matplotlib
fig, axes = plt.subplots(2, 5, figsize=(15, 6))
fig.suptitle('Noise Addition and Median Filtering', fontsize=16)

# Display original and noisy images
for idx, (name, img) in enumerate(noisy_images.items()):
    ax = axes[0, idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(name)
    ax.axis('off')

# Display median-filtered images
for idx, (name, img) in enumerate(filtered_images.items()):
    ax = axes[1, idx]
    ax.imshow(img, cmap='gray')
    ax.set_title(f"Filtered: {name}")
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(top=0.88)
plt.show()
