import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from vectorObject import Surface, VectorObject


@dataclass
class Ray:
    origin: np.ndarray  # (3,) array representing ray origin
    direction: np.ndarray  # (3,) array representing ray direction (normalized)


@dataclass
class Light:
    position: np.ndarray  # (3,) array for light position
    intensity: float  # light intensity
    color: np.ndarray  # (3,) array for RGB color


@dataclass
class Intersection:
    point: np.ndarray  # (3,) array for intersection point
    normal: np.ndarray  # (3,) array for surface normal at intersection
    distance: float  # distance from ray origin to intersection point
    surface: Surface  # reference to intersected surface


class RayTracer:
    def __init__(self, camera_position: np.ndarray, lights: List[Light]):
        self.camera_position = camera_position
        self.lights = lights
        self.ambient_intensity = 0.1
        self.specular_power = 32
        self.specular_intensity = 0.5

    def ray_triangle_intersection(
        self, ray: Ray, vertices: np.ndarray, normal: np.ndarray
    ) -> Optional[float]:
        """
        Compute ray-triangle intersection using Möller–Trumbore algorithm.
        Returns distance to intersection or None if no intersection.
        """
        v0, v1, v2 = vertices
        edge1 = v1 - v0
        edge2 = v2 - v0

        h = np.cross(ray.direction, edge2)
        a = np.dot(edge1, h)

        if abs(a) < 1e-6:  # Ray parallel to triangle
            return None

        f = 1.0 / a
        s = ray.origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None

        q = np.cross(s, edge1)
        v = f * np.dot(ray.direction, q)

        if v < 0.0 or u + v > 1.0:
            return None

        t = f * np.dot(edge2, q)

        return t if t > 1e-6 else None

    def find_intersection(
        self, ray: Ray, vector_object: VectorObject
    ) -> Optional[Intersection]:
        """Find the closest intersection between a ray and any surface in the vector object."""
        closest_intersection = None
        min_distance = float("inf")

        for surface in vector_object.surfaces:
            for face_idx in range(len(surface.faces)):
                face = surface.faces[face_idx]
                vertices = surface.vertices[face]
                normal = surface.normals[face[0]]

                distance = self.ray_triangle_intersection(ray, vertices, normal)

                if distance is not None and distance < min_distance:
                    min_distance = distance
                    intersection_point = ray.origin + ray.direction * distance
                    closest_intersection = Intersection(
                        point=intersection_point,
                        normal=normal,
                        distance=distance,
                        surface=surface,
                    )

        return closest_intersection

    def compute_lighting(self, intersection: Intersection) -> np.ndarray:
        """Compute the color at an intersection point considering all lights."""
        final_color = intersection.surface.base_color * self.ambient_intensity

        for light in self.lights:
            light_dir = light.position - intersection.point
            light_distance = np.linalg.norm(light_dir)
            light_dir /= light_distance

            view_dir = self.camera_position - intersection.point
            view_dir /= np.linalg.norm(view_dir)

            diffuse = max(0.0, np.dot(intersection.normal, light_dir))
            diffuse_color = (
                intersection.surface.base_color
                * light.color
                * (diffuse * light.intensity / light_distance**2)
            )

            # Specular lighting
            reflect_dir = (
                -light_dir
                + 2.0 * np.dot(light_dir, intersection.normal) * intersection.normal
            )
            specular = max(0.0, np.dot(view_dir, reflect_dir)) ** self.specular_power
            specular_color = light.color * (
                specular * self.specular_intensity * light.intensity / light_distance**2
            )

            final_color += diffuse_color + specular_color

        return np.clip(final_color, 0, 1)

    def trace_ray(self, ray: Ray, vector_object: VectorObject) -> np.ndarray:
        """Trace a single ray and return the color."""
        intersection = self.find_intersection(ray, vector_object)

        if intersection is None:
            return np.zeros(3)  # Background color (black)

        return self.compute_lighting(intersection)

    def render_point(
        self, point: np.ndarray, vector_object: VectorObject
    ) -> Optional[np.ndarray]:
        """
        Render a specific 3D point on the vector object.
        Returns the color at that point or None if the point isn't on any surface.
        """
        direction = point - self.camera_position
        direction /= np.linalg.norm(direction)
        ray = Ray(origin=self.camera_position, direction=direction)

        intersection = self.find_intersection(ray, vector_object)
        if intersection is None:
            return None

        if np.linalg.norm(intersection.point - point) < 1e-6:
            return self.compute_lighting(intersection)

        return None
