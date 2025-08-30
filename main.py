import pyvista as pv
import numpy as np
import trimesh
from trimesh.collision import CollisionManager

mesh1 = pv.read("")
mesh2 = pv.read("")
mesh3 = pv.read("")


def rotation(angle, axis):
    rad = np.deg2rad(angle)
    if axis == 'x':
        R = np.array([
            [1, 0, 0],
            [0, np.cos(rad), -np.sin(rad)],
            [0, np.sin(rad), np.cos(rad)]
        ])
    elif axis == 'y':
        R = np.array([
            [np.cos(rad), 0, np.sin(rad)],
            [0, 1, 0],
            [-np.sin(rad), 0, np.cos(rad)]
        ])
    elif axis == 'z':
        R = np.array([
            [np.cos(rad), -np.sin(rad), 0],
            [np.sin(rad), np.cos(rad), 0],
            [0, 0, 1]
        ])
    else:
        raise ValueError("Axis must be 'x', 'y' or 'z'")

    return R


def polydata_to_trimesh(pv_mesh: pv.PolyData) -> trimesh.Trimesh:
    faces = pv_mesh.faces.reshape((-1, 4))[:, 1:]
    vertices = np.array(pv_mesh.points)
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


local_system = np.identity(3)
origin = np.array([-58.95, -5.22, 43.25])
origins = np.tile(origin, (3, 1))
manager = CollisionManager()

i = np.linspace(35, 45, 6)
av = np.linspace(20, 30, 6)

inklinace = np.concatenate(([30], i, [50]))
anteverze_a = np.concatenate(([15], av))
anteverze_f = np.linspace(8, 17, 4)

x_vals, y_vals, z_vals, result_vals, abdukce, extenze, flexe, external_rot, internal_rot = [], [], [], [], [], [], [], [], []

for i_x, x in enumerate(inklinace):
    for i_y, y in enumerate(anteverze_a):
        for i_z, z in enumerate(anteverze_f):
            print(f"I:{x}° AV:{y}° FNV:{z}°")

            mesh2_copy = mesh2.copy()
            mesh3_copy = mesh3.copy()

            rotated_system1a = rotation(-x, 'y') @ local_system
            rotated_system2a = rotation(y, 'x') @ rotated_system1a
            rotated_system1b = rotation(-z, 'z') @ local_system

            rotated_mesh2_y = mesh2_copy.rotate_vector(local_system[1], x, point=origin)
            rotated_mesh2_x = rotated_mesh2_y.rotate_vector(rotated_system1a[0], -y, point=origin)
            rotated_mesh3_z = mesh3_copy.rotate_vector(local_system[2], z, point=origin)

            pv_mesh1 = rotated_mesh3_z
            pv_mesh2 = rotated_mesh2_x

            mesh1_new = polydata_to_trimesh(pv_mesh1)
            mesh2_new = polydata_to_trimesh(pv_mesh2)

            manager = CollisionManager()
            manager.add_object("mesh2_new", mesh2_new)
            original_mesh = mesh1_new


            def find_collision_angle(axis_vector, direction):

                if np.array_equal(axis_vector, rotated_system1b[1]) and direction == 1:
                    angle = 43 * direction
                elif np.array_equal(axis_vector, rotated_system1b[0]) and direction == 1:
                    angle = 13 * direction
                elif np.array_equal(axis_vector, rotated_system1b[0]) and direction == -1:
                    angle = 135 * direction
                elif np.array_equal(axis_vector, rotated_system1b[2]) and direction == -1:
                    angle = 32 * direction
                else:
                    angle = 36 * direction

                rotated = original_mesh.copy()
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.radians(angle),
                    axis_vector,
                    origin
                )
                rotated.apply_transform(rotation_matrix)
                base_angle = angle
                da = 1
                max_iter = 360

                while manager.in_collision_single(rotated) and da < max_iter:
                    angle = base_angle - da * direction
                    rotated = original_mesh.copy()  # reset
                    rotation_matrix = trimesh.transformations.rotation_matrix(
                        np.radians(angle),
                        axis_vector,
                        origin
                    )
                    rotated.apply_transform(rotation_matrix)
                    da += 1
                return angle


            a1 = find_collision_angle(rotated_system1b[1], 1)
            a2 = find_collision_angle(rotated_system1b[0], 1)
            a3 = find_collision_angle(rotated_system1b[0], -1)
            a4 = find_collision_angle(rotated_system1b[2], -1)
            a5 = find_collision_angle(rotated_system1b[2], 1)

            print(
                f"Abdukction:{a1}° Extension:{a2}° Felxion:{abs(a3)}° External rot.:{abs(a4)}° Internal rot.:{abs(a5)}°")

            x_vals.append(x)
            y_vals.append(y)
            z_vals.append(z)
            abdukce.append(a1)
            extenze.append(a2)
            flexe.append(abs(a3))
            external_rot.append(abs(a4))
            internal_rot.append(abs(a5))

import pandas as pd

df = pd.DataFrame({
    "Inklinace": x_vals,
    "Anteverze_A": y_vals,
    "Anteverze_F": z_vals,
    'Abdukce': abdukce,
    'Extenze': extenze,
    'Flexe': flexe,
    'External rot.': external_rot,
    'Internal rot.': internal_rot
})

df.to_csv('out.csv')

import seaborn as sns
import matplotlib.pyplot as plt

features = ['Inklinace', 'Anteverze_A', 'Anteverze_F']
targets = ['Abdukce', 'Extenze', 'Flexe']

corr = df[features + targets].corr()
corr_subset = corr.loc[features, targets]

sns.heatmap(corr_subset, annot=True, cmap='coolwarm', center=0)
plt.show()


