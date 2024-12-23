import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def generate_gaussian_light(x, y, z, light_pos, sigma, intensity):
    """
    Generate Gaussian light influence at point (x, y, z) from light source at light_pos
    with spread (sigma) and intensity.
    """
    dx, dy, dz = x - light_pos[0], y - light_pos[1], z - light_pos[2]
    distance_sq = dx**2 + dy**2 + dz**2
    return intensity * np.exp(-distance_sq / (2 * sigma**2))


def apply_gaussian_lighting_to_sphere(radius, light_sources, sigma=0.1, intensity=1):
    """
    Apply Gaussian splatting lighting to the surface of a sphere.

    Parameters:
        - radius: Radius of the sphere.
        - light_sources: List of light sources, each defined as (x, y, z, intensity).
        - sigma: Standard deviation (spread) of the Gaussian splats.
        - intensity: Base intensity of the Gaussian splats.

    Returns:
        - Arrays of points (x, y, z) on the sphere's surface, and the resulting colors.
    """
    # Create a mesh of spherical coordinates
    phi, theta = np.mgrid[
        0 : 2 * np.pi : 200j, 0 : np.pi : 100j
    ]  # Azimuthal and polar angles
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Initialize color array (white sphere initially)
    colors = np.ones_like(x)  # RGB intensity in [0, 1]

    # Apply Gaussian splats (lighting) on the sphere
    for light_x, light_y, light_z, light_intensity in light_sources:
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                # Compute the Gaussian lighting at each surface point
                light_contribution = generate_gaussian_light(
                    x[i, j],
                    y[i, j],
                    z[i, j],
                    (light_x, light_y, light_z),
                    sigma,
                    light_intensity,
                )
                colors[i, j] += light_contribution

    # Normalize color values to ensure they stay in the [0, 1] range
    colors = np.clip(colors, 0, 1)

    return x, y, z, colors


# Parameters
radius = 1  # Radius of the sphere
light_sources = [
    (1.5, 1.5, 1.5, 1),  # Light source at (1.5, 1.5, 1.5) with intensity 1
    (-1.5, -1.5, 1, 0.7),  # Light source at (-1.5, -1.5, 1) with intensity 0.7
]

# Apply Gaussian splatting lighting to the sphere
x, y, z, colors = apply_gaussian_lighting_to_sphere(
    radius, light_sources, sigma=0.2, intensity=1
)

# Visualize the sphere with lighting applied
fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection="3d")

# Plot the surface with color modifications
ax.plot_surface(
    x,
    y,
    z,
    facecolors=plt.cm.viridis(colors),
    rstride=5,
    cstride=5,
    alpha=0.7,
    antialiased=True,
)
ax.set_title("Sphere with Gaussian Splatting Lighting")
plt.show()
