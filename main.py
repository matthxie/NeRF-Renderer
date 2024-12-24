import numpy as np
import matplotlib.pyplot as plt
from vectorObject import VectorObject, Surface
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


def main():
    model = ObjectReflectanceNeRF()
    object_3d = create_example_object()
    renderer = Renderer(model, object_3d)

    camera_pos = np.array([3, 3, 3])
    light_pos = np.array([5, 5, 5])

    image = renderer.render_view(camera_pos, light_pos)

    print("Rendered image shape:", image.shape)

    plt.imshow(image)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
