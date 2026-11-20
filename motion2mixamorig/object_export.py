"""Prepare a reconstructed object and export its camera-relative animation."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
from PIL import Image

from .diagnose import read_glb
from .mixamo.gltf import _Bin, _diffuse_atlas, _quat_xyzw, _write_binary_gltf

CAMERA_TO_Y_UP = np.diag([1.0, -1.0, -1.0])


def prepare_object_mesh(source: Path, directory: Path, longest_side_m: float) -> tuple[Path, Path]:
    """Scale TripoSR's vertex-colored mesh and make matching textured OBJ and GLB."""
    import trimesh

    size = float(longest_side_m)
    if not np.isfinite(size) or size <= 0:
        raise ValueError("Object size must be a positive, finite length in metres.")
    scene = trimesh.load(str(source), force="scene", process=False)
    mesh = scene.to_geometry()
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError("TripoSR did not produce a triangle mesh.")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not np.isfinite(vertices).all() or faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Reconstructed mesh has invalid vertices or faces.")
    extent = np.ptp(vertices, axis=0).max()
    if extent <= 1e-9:
        raise ValueError("Reconstructed mesh has zero size.")
    vertices = (vertices - (vertices.min(axis=0) + vertices.max(axis=0)) / 2) * (size / extent)

    if mesh.visual.kind == "vertex":
        colors = np.asarray(mesh.visual.vertex_colors, dtype=np.uint8)[faces, :3].mean(axis=1)
    elif mesh.visual.kind == "face":
        colors = np.asarray(mesh.visual.face_colors, dtype=np.uint8)[:, :3]
    else:
        raise ValueError("TripoSR mesh is missing its vertex colors.")
    png, uv = _diffuse_atlas(colors[:, ::-1])
    # Trimesh expects bottom-up V coordinates and flips them when exporting glTF.
    uv[:, 1] = 1.0 - uv[:, 1]
    corners = vertices[faces.reshape(-1)]
    textured = trimesh.Trimesh(
        vertices=corners,
        faces=np.arange(len(corners)).reshape(-1, 3),
        vertex_normals=np.asarray(mesh.vertex_normals)[faces.reshape(-1)],
        visual=trimesh.visual.texture.TextureVisuals(
            uv=uv, image=Image.open(io.BytesIO(png)).copy()
        ),
        process=False,
    )
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    obj = directory / "object.obj"
    obj_text, textures = trimesh.exchange.obj.export_obj(
        textured, include_texture=True, return_texture=True
    )
    obj.write_text(obj_text, encoding="utf-8")
    for name, contents in textures.items():
        (directory / name).write_bytes(contents)
    static_glb = directory / "object.glb"
    static_glb.write_bytes(textured.export(file_type="glb"))
    return obj, static_glb


def write_object_glb(
    source: Path,
    destination: Path,
    timestamps: np.ndarray,
    camera_from_object: np.ndarray,
    valid: np.ndarray,
) -> Path:
    """Animate in Y-up camera coordinates; missing frames are hidden, not invented."""
    times = np.asarray(timestamps, dtype=np.float64)
    poses = np.asarray(camera_from_object, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if times.ndim != 1 or len(times) == 0 or not np.isfinite(times).all():
        raise ValueError("Object timestamps must be a non-empty finite array.")
    if times[0] < 0 or np.any(np.diff(times) <= 0):
        raise ValueError("Object timestamps must be non-negative and strictly increasing.")
    if np.any(np.diff(times.astype(np.float32)) <= 0):
        raise ValueError("Object timestamps are too close to represent in glTF.")
    if poses.shape != (len(times), 4, 4) or valid.shape != (len(times),):
        raise ValueError("Object poses and validity must match the timestamp count.")
    if not valid.any():
        raise ValueError("Object tracking produced no valid poses.")
    rotations = poses[valid, :3, :3]
    if (
        not np.isfinite(poses[valid]).all()
        or not np.allclose(poses[valid, 3], [0, 0, 0, 1], atol=1e-4)
        or not np.allclose(rotations.transpose(0, 2, 1) @ rotations, np.eye(3), atol=1e-3)
        or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-3)
    ):
        raise ValueError("Object poses contain an invalid rigid rotation or transform.")

    translations = np.zeros((len(times), 3), dtype=np.float32)
    quaternions = np.zeros((len(times), 4), dtype=np.float32)
    first = int(np.flatnonzero(valid)[0])
    previous_t = CAMERA_TO_Y_UP @ poses[first, :3, 3]
    previous_q = _quat_xyzw(CAMERA_TO_Y_UP @ poses[first, :3, :3])
    for index in range(len(times)):
        if valid[index]:
            previous_t = CAMERA_TO_Y_UP @ poses[index, :3, 3]
            q = _quat_xyzw(CAMERA_TO_Y_UP @ poses[index, :3, :3])
            if np.dot(previous_q, q) < 0:
                q = -q
            previous_q = q
        translations[index], quaternions[index] = previous_t, previous_q

    gltf, binary = read_glb(source)
    if len(gltf.get("buffers", [])) != 1:
        raise ValueError("Object GLB must contain one embedded buffer.")
    scene_index = gltf.get("scene", 0)
    scene = gltf["scenes"][scene_index]
    node_index = len(gltf["nodes"])
    gltf["nodes"].append({"name": "Object camera-relative motion", "children": scene["nodes"]})
    scene["nodes"] = [node_index]
    blob = _Bin()
    blob.add(binary)

    def accessor(values: np.ndarray, kind: str) -> int:
        values = np.asarray(values, dtype="<f4")
        offset, length = blob.add_array(values)
        views = gltf.setdefault("bufferViews", [])
        views.append({"buffer": 0, "byteOffset": offset, "byteLength": length})
        flat = values.reshape(len(values), -1)
        accessors = gltf.setdefault("accessors", [])
        accessors.append({
            "bufferView": len(views) - 1,
            "componentType": 5126,
            "count": len(values),
            "type": kind,
            "min": flat.min(axis=0).tolist(),
            "max": flat.max(axis=0).tolist(),
        })
        return len(accessors) - 1

    time_accessor = accessor(times, "SCALAR")
    animation = {"name": "Object motion", "samplers": [], "channels": []}
    for name, values, kind, interpolation in (
        ("translation", translations, "VEC3", "LINEAR"),
        ("rotation", quaternions, "VEC4", "LINEAR"),
        ("scale", np.repeat(valid[:, None], 3, axis=1), "VEC3", "STEP"),
    ):
        output = accessor(values, kind)
        sampler = len(animation["samplers"])
        animation["samplers"].append({
            "input": time_accessor, "output": output, "interpolation": interpolation,
        })
        animation["channels"].append({
            "sampler": sampler, "target": {"node": node_index, "path": name},
        })
    gltf["animations"] = [animation]
    data = blob.dumps()
    gltf["buffers"][0]["byteLength"] = len(data)
    return _write_binary_gltf(destination, gltf, data)
