"""Original low-poly T-pose mannequin; no downloaded character assets."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .kinematics import MixamoBone, MixamoRig
from .skinned_mesh import SkinnedMesh, YBotAsset

NAME = "4onstudios_default_mannequin"


def build_mannequin(source: Path) -> YBotAsset:
    """Create a centimetre-scale Mixamo-compatible rig and rigid-skinned doll."""
    joints: list[tuple[str, str | None, tuple[float, float, float]]] = [
        ("Hips", None, (0, 100, 0)),
        ("Spine", "Hips", (0, 112, 0)),
        ("Spine1", "Spine", (0, 124, 0)),
        ("Spine2", "Spine1", (0, 138, 0)),
        ("Neck", "Spine2", (0, 151, 0)),
        ("Head", "Neck", (0, 163, 0)),
        ("HeadTop_End", "Head", (0, 180, 0)),
    ]
    for side, sign in (("Left", 1), ("Right", -1)):
        joints.extend([
            (f"{side}Shoulder", "Spine2", (sign * 8, 142, 0)),
            (f"{side}Arm", f"{side}Shoulder", (sign * 20, 142, 0)),
            (f"{side}ForeArm", f"{side}Arm", (sign * 49, 142, 0)),
            (f"{side}Hand", f"{side}ForeArm", (sign * 74, 142, 0)),
            (f"{side}Hand_End", f"{side}Hand", (sign * 86, 142, 0)),
            (f"{side}UpLeg", "Hips", (sign * 10, 95, 0)),
            (f"{side}Leg", f"{side}UpLeg", (sign * 10, 53, 0)),
            (f"{side}Foot", f"{side}Leg", (sign * 10, 10, 0)),
            (f"{side}ToeBase", f"{side}Foot", (sign * 10, 10, 15)),
            (f"{side}Toe_End", f"{side}ToeBase", (sign * 10, 10, 24)),
        ])
    bones: dict[str, MixamoBone] = {}
    for suffix, parent_suffix, position in joints:
        name = f"mixamorig:{suffix}"
        parent = None if parent_suffix is None else f"mixamorig:{parent_suffix}"
        world = np.eye(4)
        world[:3, 3] = position
        local = world.copy() if parent is None else np.linalg.inv(bones[parent].rest_world) @ world
        bones[name] = MixamoBone(
            name=name, parent=parent, children=[], rest_rotation=np.eye(3),
            lcl_translation_cm=local[:3, 3].copy(), rest_world=world,
            rest_local=local, pre_rotation_euler_deg=np.zeros(3),
        )
        if parent is not None:
            bones[parent].children.append(name)
    order = list(bones)
    rig = MixamoRig(bones, order, 0.01, Path(source))
    bone_index = {name: index for index, name in enumerate(order)}
    ibm = np.stack([np.linalg.inv(bones[name].rest_world) for name in order])
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    influences: list[np.ndarray] = []
    corners = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1],
    ])
    triangles = np.array([
        [0, 2, 1], [0, 3, 2], [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4], [3, 7, 6], [3, 6, 2],
        [0, 4, 7], [0, 7, 3], [1, 2, 6], [1, 6, 5],
    ], dtype=np.int32)

    def box(bone: str, center: np.ndarray, half_size: tuple[float, float, float]) -> None:
        offset = len(vertices) * 8
        vertices.append(center + corners * half_size)
        faces.append(triangles + offset)
        weights = np.zeros((8, len(order)))
        weights[:, bone_index[f"mixamorig:{bone}"]] = 1
        influences.append(weights)

    def segment(bone: str, child: str, width: float, depth: float) -> None:
        start = bones[f"mixamorig:{bone}"].rest_world[:3, 3]
        end = bones[f"mixamorig:{child}"].rest_world[:3, 3]
        half = np.abs(end - start) / 2
        axis = int(np.argmax(half))
        half[axis] *= 0.88
        if axis == 0:
            half[1:] = (width, depth)
        else:
            half[0], half[2] = width, depth
        box(bone, (start + end) / 2, (half[0], half[1], half[2]))

    box("Hips", np.array([0, 100, 0]), (17, 7, 9))
    segment("Spine", "Spine1", 15, 8)
    segment("Spine1", "Spine2", 18, 9)
    segment("Spine2", "Neck", 18, 9)
    segment("Neck", "Head", 4, 4)
    box("Head", np.array([0, 171, 0]), (9, 10, 8))
    for side, sign in (("Left", 1), ("Right", -1)):
        segment(f"{side}Arm", f"{side}ForeArm", 5, 5)
        segment(f"{side}ForeArm", f"{side}Hand", 4, 4)
        segment(f"{side}Hand", f"{side}Hand_End", 4, 2.5)
        segment(f"{side}UpLeg", f"{side}Leg", 7, 7)
        segment(f"{side}Leg", f"{side}Foot", 5, 5)
        box(f"{side}Foot", np.array([sign * 10, 5, 5]), (6, 5, 10))
        box(f"{side}ToeBase", np.array([sign * 10, 5, 20]), (6, 5, 4))

    mesh = SkinnedMesh(
        name=NAME, vertices=np.concatenate(vertices), faces=np.concatenate(faces),
        weights=np.concatenate(influences), ibm=ibm, color=np.array([0.25, 0.55, 0.85]),
    )
    return YBotAsset(rig, [mesh], bone_index)
