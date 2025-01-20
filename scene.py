import numpy as np
from typing import List, Tuple
from dataclasses import dataclass
import math
import matplotlib.pyplot as plt


@dataclass
class Vector2D:
    x: float
    y: float

    def __add__(self, other):
        return Vector2D(self.x + other.x, self.y + other.y)

    def __sub__(self, other):
        return Vector2D(self.x - other.x, self.y - other.y)

    def dot(self, other):
        return self.x * other.x + self.y * other.y

    def length(self):
        return math.sqrt(self.x * self.x + self.y * self.y)

    def normalize(self):
        length = self.length()
        if length == 0:
            return Vector2D(0, 0)
        return Vector2D(self.x / length, self.y / length)


@dataclass
class Circle:
    position: Vector2D
    radius: float
    color: Tuple[float, float, float]

    def intersect(self, origin: Vector2D, direction: Vector2D) -> float:
        oc = origin - self.position
        a = direction.dot(direction)
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - self.radius * self.radius
        discriminant = b * b - 4 * a * c

        if abs(a) < 1e-10:
            return float("inf")

        if discriminant < 0:
            return float("inf")

        t = (-b - math.sqrt(discriminant)) / (2.0 * a)
        return t if t > 0 else float("inf")


@dataclass
class LightSource:
    position: Vector2D
    intensity: float


class Scene:
    def __init__(
        self, width: int, height: int, camera_position: Vector2D = Vector2D(0, 0)
    ):
        self.width = width
        self.height = height
        self.camera_position = camera_position
        self.objects: List[Circle] = []
        self.lights: List[LightSource] = []
        self.pixels = np.zeros((height, width, 3))

    def add_object(self, obj: Circle):
        self.objects.append(obj)

    def add_light(self, light: LightSource):
        self.lights.append(light)

    def trace_ray(self, origin: Vector2D, direction: Vector2D) -> Tuple[float, Circle]:
        closest_t = float("inf")
        closest_obj = None

        for obj in self.objects:
            t = obj.intersect(origin, direction)
            if t < closest_t:
                closest_t = t
                closest_obj = obj

        return closest_t, closest_obj

    def calculate_lighting(self, point: Vector2D, obj: Circle) -> float:
        total_intensity = 0.0

        for light in self.lights:
            light_dir = (light.position - point).normalize()
            shadow_origin = point
            _, shadow_obj = self.trace_ray(shadow_origin, light_dir)

            if shadow_obj is None:
                normal = (point - obj.position).normalize()
                intensity = max(0, light_dir.dot(normal))
                total_intensity += intensity * light.intensity

        return min(1.0, total_intensity)

    def render(self):
        for y in range(self.height):
            for x in range(self.width):
                scene_x = (x - self.width / 2) / (self.width / 2)
                scene_y = (self.height / 2 - y) / (self.height / 2)

                pixel_pos = Vector2D(scene_x, scene_y)
                direction = (pixel_pos - self.camera_position).normalize()

                t, obj = self.trace_ray(self.camera_position, direction)

                if obj is not None:
                    intersection = Vector2D(
                        self.camera_position.x + direction.x * t,
                        self.camera_position.y + direction.y * t,
                    )

                    light_intensity = self.calculate_lighting(intersection, obj)

                    self.pixels[y, x] = [
                        obj.color[0] * light_intensity,
                        obj.color[1] * light_intensity,
                        obj.color[2] * light_intensity,
                    ]

        plt.imshow(self.pixels)
        plt.axis("off")
        plt.show()
