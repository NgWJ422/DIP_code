import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 40  # Resolution (higher number = smoother cube)

# Generate RGB values
r = np.linspace(0, 1, N)
g = np.linspace(0, 1, N)
b = np.linspace(0, 1, N)
R, G, B = np.meshgrid(r, g, b)

# Reshape for 3D scatter plot
R_flat = R.flatten()
G_flat = G.flatten()
B_flat = B.flatten()

# Plot RGB cube
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = np.stack((R_flat, G_flat, B_flat), axis=1)  # Combine RGB values for color mapping

scatter = ax.scatter(R_flat, G_flat, B_flat, c=colors, marker='o')
ax.set_xlabel('Red (R)')
ax.set_ylabel('Green (G)')
ax.set_zlabel('Blue (B)')
ax.set_title('RGB Color Cube')

# Adjust the view for better visualization
ax.view_init(elev=30, azim=-135)
plt.grid(True)
plt.show()
