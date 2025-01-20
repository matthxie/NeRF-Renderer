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
            points = surface.sample_surface_points(80)

            view_dirs = camera_position - points
            view_dirs /= np.linalg.norm(view_dirs, axis=-1, keepdims=True)

            light_dirs = light_position - points
            light_dirs /= np.linalg.norm(light_dirs, axis=-1, keepdims=True)

            positions = torch.FloatTensor(points)
            normals = torch.FloatTensor(self._interpolate_normals(surface, points))
            view_dirs = torch.FloatTensor(view_dirs)
            light_dirs = torch.FloatTensor(light_dirs)
            base_colors = torch.FloatTensor(surface.base_color).expand(len(points), -1)

            with torch.no_grad():
                modified_colors = self.model(
                    positions, view_dirs, light_dirs, base_colors
                ).numpy()

            self._update_render_buffer(
                render_buffer, points, modified_colors, camera_position, resolution
            )

        return render_buffer

    def render_scene(
        self,
        camera_position: np.ndarray,
        lighting: np.ndarray,
        resolution: Tuple[int, int] = (512, 512),
    ) -> np.ndarray:
        height, width = resolution
        render_buffer = np.zeros((height, width, 3))

        visible_surfaces = self.object_3d.get_visible_surfaces(camera_position)

    def set_colours(self, colours):
        pass

    def _interpolate_normals(self, surface: Surface, points: np.ndarray) -> np.ndarray:
        avg_normal = surface.normals.mean(axis=0)
        return np.tile(avg_normal, (len(points), 1))

    def _update_render_buffer(
        self,
        buffer: np.ndarray,
        points: np.ndarray,
        colors: np.ndarray,
        camera_position: np.ndarray,
        resolution: Tuple[int, int],
    ):
        height, width = resolution

        f = 100
        for point, color in zip(points, colors):
            d = point - camera_position
            x = int(f * d[0] / d[2] + width / 2)
            y = int(f * d[1] / d[2] + height / 2)

            if 0 <= x < width and 0 <= y < height:
                buffer[y, x] = color
