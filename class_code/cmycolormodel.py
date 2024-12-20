import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Parameters
N = 40  # Resolution (higher number = smoother cube)

# Generate CMY values
c = np.linspace(0, 1, N)
m = np.linspace(0, 1, N)
y = np.linspace(0, 1, N)
C, M, Y = np.meshgrid(c, m, y)

# Convert CMY to RGB
R = 1 - C
G = 1 - M
B = 1 - Y

# Reshape for 3D scatter plot
C_flat = C.flatten()
M_flat = M.flatten()
Y_flat = Y.flatten()
RGB_flat = np.stack((R.flatten(), G.flatten(), B.flatten()), axis=1)

# Plot CMY Cube
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(C_flat, M_flat, Y_flat, c=RGB_flat, marker='o')
ax.set_xlabel('Cyan (C)')
ax.set_ylabel('Magenta (M)')
ax.set_zlabel('Yellow (Y)')
ax.set_title('CMY Color Cube')

# Adjust the view for better visualization
ax.view_init(elev=30, azim=135)
plt.grid(True)
plt.show()
