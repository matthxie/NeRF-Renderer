import numpy as np
from scene import Scene, Circle, Vector2D, LightSource
from objectReflectanceNerf import ObjectReflectanceNeRF


def main():
    model = ObjectReflectanceNeRF()
    scene = Scene(400, 400)

    scene.add_object(Circle(Vector2D(0.0, 0.0), 0.2, (1.0, 0.0, 0.0)))  # Red circle
    scene.add_object(Circle(Vector2D(0.3, 0.3), 0.1, (0.0, 1.0, 0.0)))  # Green circle

    scene.add_light(LightSource(Vector2D(0.5, 0.5), 1.0))

    scene.render()

    # Move an object and re-render
    scene.objects[0].position = Vector2D(-0.2, 0.1)
    scene.render()


if __name__ == "__main__":
    main()
