import open3d as o3d
import numpy as np

mesh = o3d.geometry.TriangleMesh.create_sphere(radius=1.0)
mesh.compute_vertex_normals()

# Set material properties
mesh.paint_uniform_color([1.0, 0.5, 0.0])  # Orange color

# Create a visualizer and add the mesh
vis = o3d.visualization.Visualizer()
vis.create_window()
vis.add_geometry(mesh)

# Run the visualizer
vis.run()
vis.destroy_window()
