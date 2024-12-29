import numpy as np
import matplotlib.pyplot as plt
from vectorObject import VectorObject, Surface
from rayTracer import RayTracer, Light
from objectReflectanceNerf import ObjectReflectanceNeRF
from render import Renderer


def create_example_object() -> VectorObject:
    vertices = np.array(
        [
            [-1, -1, -1],
            [1, -1, -1],
            [1, 1, -1],
            [-1, 1, -1],
            [-1, -1, 1],
            [1, -1, 1],
            [1, 1, 1],
            [-1, 1, 1],
        ]
    )

    normals = np.array(
        [
            [0, 0, -1],
            [0, 0, 1],  # front, back
            [1, 0, 0],
            [-1, 0, 0],  # right, left
            [0, 1, 0],
            [0, -1, 0],  # top, bottom
        ]
    )

    surfaces = []
    colors = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    faces = [
        [0, 1, 2, 3],  # front
        [4, 5, 6, 7],  # back
        [1, 5, 6, 2],  # right
        [0, 4, 7, 3],  # left
        [3, 2, 6, 7],  # top
        [0, 1, 5, 4],  # bottom
    ]

    for i, (face, normal, color) in enumerate(zip(faces, normals, colors)):
        face_vertices = vertices[face]
        face_normals = np.tile(normal, (len(face_vertices), 1))
        face_indices = np.array([[0, 1, 2], [0, 2, 3]])

        surfaces.append(
            Surface(
                vertices=face_vertices,
                normals=face_normals,
                faces=face_indices,
                base_color=np.array(color),
            )
        )

    return VectorObject(surfaces)


def create_example_dataset(
    init_camera_position: np.ndarray,
    init_light_position: np.ndarray,
    object: VectorObject,
):
    light = Light(
        position=init_light_position, intensity=1.0, color=np.array([1, 1, 1])
    )
    camera_position = init_camera_position

    ray_tracer = RayTracer(init_camera_position, [light])

    positions = []
    view_dirs = []
    light_dirs = []
    colours = []

    for surface, _ in object.get_visible_surfaces(camera_position):
        points = surface.sample_surface_points(20)

        for point in points:
            colour = ray_tracer.render_point(point, object)

            cam_diff = camera_position - point
            light_diff = light.position - point

            theta_cam = np.degrees(np.arctan2(cam_diff[1], cam_diff[0]))
            phi_cam = np.degrees(
                np.arctan2(cam_diff[2], np.sqrt(cam_diff[0] ** 2 + cam_diff[1] ** 2))
            )

            theta_light = np.degrees(np.arctan2(light_diff[1], light_diff[0]))
            phi_light = np.degrees(
                np.arctan2(
                    light_diff[2], np.sqrt(light_diff[0] ** 2 + light_diff[1] ** 2)
                )
            )

            positions.append(point)
            view_dirs.append(np.array([theta_cam, phi_cam]))
            light_dirs.append(np.array([theta_light, phi_light]))
            colours.append(colour)

    return positions, view_dirs, light_dirs, colours


def main():
    model = ObjectReflectanceNeRF()
    object_3d = create_example_object()
    renderer = Renderer(model, object_3d)

    camera_position = np.array([3, 3, 3])
    light_position = np.array([5, 5, -5])

    image = renderer.render_view(camera_position, light_position)

    print("Rendered image shape:", image.shape)

    positions, view_dirs, light_dirs, colours = create_example_dataset(
        camera_position, light_position, object_3d
    )

    print(colours)

    # plt.imshow(image)
    # plt.axis("off")
    # plt.show()


if __name__ == "__main__":
    main()
