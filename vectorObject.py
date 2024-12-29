import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Surface:
    vertices: np.ndarray  # (N, 3) array of vertex positions
    normals: np.ndarray  # (N, 3) array of vertex normals
    faces: np.ndarray  # (M, 3) array of face indices
    base_color: np.ndarray  # (3,) RGB color

    def __eq__(self, other):
        if isinstance(other, Surface):
            return (
                np.array_equal(self.vertices, other.vertices)
                and np.array_equal(self.normals, other.normals)
                and np.array_equal(self.faces, other.faces)
                and np.array_equal(self.base_color, other.base_color)
            )
        return False

    def sample_surface_points(self, num_samples: int) -> np.ndarray:
        points = []
        for face in self.faces:
            v1, v2, v3 = self.vertices[face]
            for _ in range(num_samples):
                a, b = np.random.random(2)
                if a + b > 1:
                    a, b = 1 - a, 1 - b
                c = 1 - a - b
                point = a * v1 + b * v2 + c * v3
                points.append(point)
        return np.array(points)


class VectorObject:
    def __init__(self, surfaces: List[Surface]):
        self.surfaces = surfaces

    def get_visible_surfaces(
        self, camera_position: np.ndarray
    ) -> List[Tuple[Surface, np.ndarray]]:
        visible = []
        for surface in self.surfaces:
            center = surface.vertices.mean(axis=0)

            avg_normal = surface.normals.mean(axis=0)
            avg_normal /= np.linalg.norm(avg_normal)

            # Check if surface faces camera (dot product > 0)
            view_dir = camera_position - center
            view_dir /= np.linalg.norm(view_dir)

            if np.dot(avg_normal, view_dir) > 0:
                visible.append((surface, center))

        return visible
