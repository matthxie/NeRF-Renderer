import torch
import numpy as np
from vectorObject import VectorObject, Surface
from objectReflectanceNerf import ObjectReflectanceNeRF
from typing import Tuple


class Renderer:
    def __init__(self, model: ObjectReflectanceNeRF, object_3d: VectorObject):
        self.model = model
        self.object_3d = object_3d

    def render_view(
        self,
        camera_position: np.ndarray,
        light_position: np.ndarray,
        resolution: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        height, width = resolution
        render_buffer = np.zeros((height, width, 3))

        visible_surfaces = self.object_3d.get_visible_surfaces(camera_position)

        for surface, center in visible_surfaces:
            points = self._generate_surface_points(surface, resolution)

            view_dirs = camera_position - points
            view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)

            light_dirs = light_position - points
            light_dirs /= np.linalg.norm(light_dirs, axis=-1, keepdims=True)

            positions = torch.FloatTensor(points)
            normals = torch.FloatTensor(self._interpolate_normals(surface, points))
            surface_ids = torch.FloatTensor(self._one_hot_encode_surface(surface))
            view_dirs = torch.FloatTensor(view_dirs)
            light_dirs = torch.FloatTensor(light_dirs)
            base_colors = torch.FloatTensor(surface.base_color).expand(len(points), -1)

            with torch.no_grad():
                modified_colors = self.model(positions, view_dirs, light_dirs).numpy()

            self._update_render_buffer(
                render_buffer, points, modified_colors, camera_position, resolution
            )

        return render_buffer

    def _generate_surface_points(
        self, surface: Surface, resolution: Tuple[int, int]
    ) -> np.ndarray:

        points = []
        for face in surface.faces:
            v1, v2, v3 = surface.vertices[face]
            num_samples = 10
            for _ in range(num_samples):
                a, b = np.random.random(2)
                if a + b > 1:
                    a, b = 1 - a, 1 - b
                c = 1 - a - b
                point = a * v1 + b * v2 + c * v3
                points.append(point)
        return np.array(points)

    def _interpolate_normals(self, surface: Surface, points: np.ndarray) -> np.ndarray:
        avg_normal = surface.normals.mean(axis=0)
        return np.tile(avg_normal, (len(points), 1))

    def _one_hot_encode_surface(self, surface: Surface) -> np.ndarray:
        surface_idx = self.object_3d.surfaces.index(surface)
        encoding = np.zeros(3)
        encoding[surface_idx % 3] = 1
        return np.tile(encoding, (len(surface.vertices), 1))

    def _update_render_buffer(
        self,
        buffer: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray,
        camera_position: np.ndarray,
        resolution: Tuple[int, int],
    ):
        height, width = resolution

        f = 500
        for point, color in zip(points, colors):
            d = point - camera_position
            x = int(f * d[0] / d[2] + width / 2)
            y = int(f * d[1] / d[2] + height / 2)

            if 0 <= x < width and 0 <= y < height:
                buffer[y, x] = color
