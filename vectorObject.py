import numpy as np
from typing import List, Tuple
from dataclasses import dataclass


@dataclass
class Surface:
    vertices: np.ndarray  # (N, 3) array of vertex positions
    normals: np.ndarray  # (N, 3) array of vertex normals
    faces: np.ndarray  # (M, 3) array of face indices
    base_color: np.ndarray  # (3,) RGB color


class VectorObject:
    def __init__(self, surfaces: List[Surface]):
        self.surfaces = surfaces

    def get_visible_surfaces(
        self, camera_position: np.ndarray
    ) -> List[Tuple[Surface, np.ndarray]]:
        """Return visible surfaces and their center points based on camera position"""
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
