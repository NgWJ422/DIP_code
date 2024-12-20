import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

# Define higher-resolution ranges for HSV and HSI values
H = np.linspace(0, 1, 100)  # Hue (0 to 1) with higher resolution
S = np.linspace(0, 1, 100)  # Saturation (0 to 1) with higher resolution
V = np.linspace(0, 1, 100)  # Value (0 to 1) with higher resolution
I = np.linspace(0, 1, 100)  # Intensity (0 to 1) with higher resolution

# Generate a fine grid of HSV and HSI values
H_grid, S_grid, V_grid = np.meshgrid(H, S, V)
H_grid_hsi, S_grid_hsi, I_grid = np.meshgrid(H, S, I)

# Convert HSV to RGB
hsv = np.stack((H_grid, S_grid, V_grid), axis=-1)
rgb_hsv = hsv_to_rgb(hsv)

# Convert HSI to RGB (as a placeholder for now, you can refine later)
hsi = np.stack((H_grid_hsi, S_grid_hsi, I_grid), axis=-1)
rgb_hsi = hsv_to_rgb(hsi)  # Since HSI is similar to HSV in conversion to RGB

# HSV Cone (upright cone)
Z_hsv = V_grid.flatten()  # Height (Value)
R_hsv = S_grid.flatten() * Z_hsv  # Radius decreases with Value (cone shape)
X_hsv = R_hsv * np.cos(2 * np.pi * H_grid.flatten())  # X-coordinate
Y_hsv = R_hsv * np.sin(2 * np.pi * H_grid.flatten())  # Y-coordinate
colors_hsv = rgb_hsv.reshape(-1, 3)  # Flatten RGB for coloring

# HSI Cone (inverted cone)
Z_hsi = -I_grid.flatten()  # Height (Intensity), negative to invert the cone
R_hsi = S_grid_hsi.flatten() * np.abs(Z_hsi)  # Radius depends on Intensity (cone shape)
X_hsi = R_hsi * np.cos(2 * np.pi * H_grid_hsi.flatten())  # X-coordinate
Y_hsi = R_hsi * np.sin(2 * np.pi * H_grid_hsi.flatten())  # Y-coordinate
colors_hsi = rgb_hsi.reshape(-1, 3)  # Flatten RGB for coloring

# Adjust the Z-values of both cones to align their bases
Z_hsi_offset = np.max(Z_hsv)  # Find the maximum Z-value of HSV
Z_hsi = Z_hsi + 2*Z_hsi_offset  # Offset HSI cone to ensure the bases meet

# Plot the combined HSV and HSI color cones
fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')

# Plot HSV (upright cone)
scatter_hsv = ax.scatter(X_hsv, Y_hsv, Z_hsv, c=colors_hsv, s=1, edgecolor='none', alpha=0.9)

# Plot HSI (inverted cone)
scatter_hsi = ax.scatter(X_hsi, Y_hsi, Z_hsi, c=colors_hsi, s=1, edgecolor='none', alpha=0.9)

# Set axis labels
ax.set_xlabel('X (Hue & Saturation)', fontsize=10)
ax.set_ylabel('Y (Hue & Saturation)', fontsize=10)
ax.set_zlabel('Z (Value/Intensity)', fontsize=10)

# Set title and adjust viewing angle
ax.set_title('Combined HSV (Upright) and HSI (Inverted) Color Cones', fontsize=14)
ax.view_init(elev=10, azim=0)  # Adjust view for a better perspective

# Display the plot
plt.show()
