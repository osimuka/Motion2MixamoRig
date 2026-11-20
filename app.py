"""Private Hugging Face Space UI for the 4onstudios LTD Motion2MixamoRig pipeline."""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
import uuid
import zipfile
from collections.abc import Callable
from pathlib import Path

SPACE_DATA_ROOT = Path(os.environ.get("M2MR_SPACE_DATA_ROOT", "/data"))
GRADIO_TEMP_DIR = Path(
    os.environ.get("GRADIO_TEMP_DIR", Path(tempfile.gettempdir()) / "m2mr-gradio")
)
GRADIO_TEMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("GRADIO_TEMP_DIR", str(GRADIO_TEMP_DIR))
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
os.environ.setdefault("M2MR_ASSETS_DIR", str(SPACE_DATA_ROOT / "assets"))
os.environ.setdefault("M2MR_WEIGHTS_DIR", str(SPACE_DATA_ROOT / "weights"))

import gradio as gr

from motion2mixamorig import checks, paths
from motion2mixamorig.errors import classify_error
from motion2mixamorig.extractors.sam3d import require_cuda
from motion2mixamorig.pipeline import run_pipeline
from motion2mixamorig.progress import ProgressWriter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("motion2mixamorig.space")

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BUNDLE_FILES = (
    "mixamo_character.glb",
    "mixamo_character.gif",
    "mixamo_rotations.npz",
    "skeleton_motion.npz",
    "diagnostics.json",
)


class SpaceProgressWriter(ProgressWriter):
    def __init__(self, progress: gr.Progress, scale: float = 1.0) -> None:
        super().__init__()
        self.progress = progress
        self.scale = scale

    def emit(self, stage: str, progress: float, message: str) -> dict:
        event = super().emit(stage, progress, message)
        self.progress(event["progress"] * self.scale, desc=event["message"])
        return event


def _copy_upload(
    upload: str | None, destination: Path, allowed: set[str], label: str
) -> Path:
    if not upload:
        raise gr.Error(f"Choose a {label} file.")
    source = Path(upload)
    if not source.is_file():
        raise gr.Error(f"The uploaded {label} is no longer available. Upload it again.")
    suffix = source.suffix.lower()
    if suffix not in allowed:
        raise gr.Error(f"Unsupported {label} format: {suffix or '(no file extension)'}.")
    target = destination / f"{label}{suffix}"
    shutil.copyfile(source, target)
    return target


def _validate_upload(
    path: Path, checker: Callable[[Path], str | None], label: str
) -> None:
    problem = checker(path)
    if problem is not None:
        raise gr.Error(f"{label} is not usable: {problem}")


def _zip_outputs(run_dir: Path, *, include_object: bool = False) -> Path:
    bundle_dir = GRADIO_TEMP_DIR / "downloads"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive = bundle_dir / f"motion2mixamorig-{uuid.uuid4().hex}.zip"
    included = [run_dir / name for name in BUNDLE_FILES if (run_dir / name).is_file()]
    if not (run_dir / "mixamo_character.glb").is_file():
        raise RuntimeError("The pipeline completed without producing mixamo_character.glb.")
    if include_object:
        from motion2mixamorig.object_tracking import OBJECT_FILES

        missing = [name for name in OBJECT_FILES if not (run_dir / name).is_file()]
        if missing:
            raise RuntimeError(f"Object capture did not produce: {', '.join(missing)}.")
        included.extend(run_dir / name for name in OBJECT_FILES)
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in included:
            bundle.write(file, arcname=file.name)
    return archive


def _video_frame(video: str | None, frame_number: int | float) -> tuple:
    import cv2

    if not video or not Path(video).is_file():
        raise gr.Error("Upload a video before selecting an object.")
    try:
        frame_index = int(frame_number)
        if frame_index != frame_number or frame_index < 0:
            raise ValueError
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error("Object frame must be a non-negative whole number.") from exc
    capture = cv2.VideoCapture(video)
    try:
        if not capture.isOpened() or not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index):
            raise gr.Error("Cannot seek to the selected video frame.")
        ok, image = capture.read()
        if not ok or abs(capture.get(cv2.CAP_PROP_POS_FRAMES) - frame_index - 1) > 0.5:
            raise gr.Error("The requested object frame is not available in this video.")
        return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), frame_index
    finally:
        capture.release()


def preview_object_frame(video: str | None, frame_number: int | float):
    image, index = _video_frame(video, frame_number)
    state = {
        "video": str(Path(video).resolve()), "frame": index,
        "width": image.shape[1], "height": image.shape[0], "points": [],
    }
    return image, state, "Click two opposite corners around the object."


def select_object_point(
    video: str | None, frame_number: int | float, state: dict | None, event: gr.SelectData,
):
    import cv2

    if not state or not video or str(Path(video).resolve()) != state["video"] or frame_number != state["frame"]:
        raise gr.Error("Load the object frame again after changing the video or frame number.")
    image, _ = _video_frame(video, frame_number)
    x, y = event.index
    x, y = int(x), int(y)
    if not (0 <= x < state["width"] and 0 <= y < state["height"]):
        raise gr.Error("Click inside the object frame.")
    points = list(state["points"])
    points = [] if len(points) == 2 else points
    points.append((x, y))
    state = {**state, "points": points}
    cv2.circle(image, (x, y), 5, (255, 170, 0), -1)
    if len(points) == 1:
        message = "Click the opposite corner."
    else:
        cv2.rectangle(image, points[0], points[1], (255, 170, 0), 2)
        message = "Object selected. Click again to start a new box."
    return image, state, message


def _object_selection(
    video: str | None, frame_number: int | float, state: dict | None, size_cm: float | None,
):
    import numpy as np
    from motion2mixamorig.object_tracking import ObjectSelection

    if (
        not state or not video or str(Path(video).resolve()) != state["video"]
        or frame_number != state["frame"] or len(state["points"]) != 2
    ):
        raise gr.Error("Load the object frame and click two opposite corners around the object.")
    try:
        size = float(size_cm)
    except (TypeError, ValueError) as exc:
        raise gr.Error("Object size must be a positive number of centimetres.") from exc
    if not np.isfinite(size) or size <= 0:
        raise gr.Error("Object size must be a positive number of centimetres.")
    (x0, y0), (x1, y1) = state["points"]
    box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
    if box[2] - box[0] < 2 or box[3] - box[1] < 2:
        raise gr.Error("The object selection is too small; select it again.")
    return ObjectSelection(state["frame"], box, state["width"], state["height"], size / 100)


def _release_cuda_memory() -> None:
    import gc
    import torch

    gc.collect()
    torch.cuda.empty_cache()


def generate(
    media_type: str,
    video_upload: str | None,
    image_upload: str | None,
    rig_upload: str | None,
    detect_hands: bool = False,
    hand_smooth_s: float = 0.08,
    track_person: bool = False,
    capture_held_object: bool = False,
    object_frame: int = 0,
    object_state: dict | None = None,
    object_size_cm: float | None = None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    uploads = (video_upload, image_upload, rig_upload)
    try:
        try:
            require_cuda("cuda")
        except RuntimeError as exc:
            raise gr.Error(str(exc)) from exc

        with tempfile.TemporaryDirectory(prefix="m2mr-space-") as temp:
            job_dir = Path(temp)
            input_dir = job_dir / "input"
            input_dir.mkdir()
            rig = None
            if rig_upload:
                rig = _copy_upload(rig_upload, input_dir, {".fbx"}, "character")
                _validate_upload(rig, checks.check_rig, "Mixamo character")
            elif os.environ.get("M2MR_DEFAULT_RIG"):
                rig = Path(os.environ["M2MR_DEFAULT_RIG"])
                if not rig.is_file():
                    raise gr.Error("The configured default character is missing. Contact the Space owner.")
                _validate_upload(rig, checks.check_rig, "Default character")

            if media_type == "Video":
                source = _copy_upload(video_upload, input_dir, VIDEO_SUFFIXES, "source")
                _validate_upload(
                    source, lambda path: checks.check_video(path, check_people=False), "Video"
                )
                video, image = source, None
            elif media_type == "Still image":
                source = _copy_upload(image_upload, input_dir, IMAGE_SUFFIXES, "source")
                _validate_upload(
                    source, lambda path: checks.check_image(path, check_people=False), "Image"
                )
                video, image = None, source
            else:
                raise gr.Error("Choose either Video or Still image.")

            if capture_held_object:
                if video is None:
                    raise gr.Error("Object capture currently supports video input only.")
                from motion2mixamorig.object_tracking import object_runtimes

                selection = _object_selection(video_upload, object_frame, object_state, object_size_cm)
                try:
                    runtimes = object_runtimes()
                except RuntimeError as exc:
                    raise gr.Error(str(exc)) from exc

            device = "cuda"
            run_dir = job_dir / "output"
            body_progress_fraction = 0.5 if capture_held_object else 1.0
            try:
                run_pipeline(
                    video,
                    rig,
                    image=image,
                    device=device,
                    extractor="sam3d",
                    detect_hands=detect_hands,
                    hand_smooth_s=hand_smooth_s,
                    track_person=track_person,
                    verbose=False,
                    output_dir=run_dir,
                    progress=SpaceProgressWriter(progress, scale=body_progress_fraction),
                    no_preview=False,
                )
            except Exception as exc:
                LOGGER.exception("Motion2MixamoRig generation failed")
                code, message = classify_error(exc)
                raise gr.Error(f"Generation failed ({code}): {message}") from exc

            if capture_held_object:
                from motion2mixamorig.object_tracking import capture_object

                try:
                    _release_cuda_memory()
                    capture_object(
                        video, run_dir / "skeleton_motion.npz", selection, run_dir, runtimes,
                        progress=lambda value, message: progress(
                            body_progress_fraction + (1 - body_progress_fraction) * value,
                            desc=message,
                        ),
                    )
                except Exception as exc:
                    LOGGER.exception("Object capture failed")
                    raise gr.Error(f"Object capture failed: {exc}") from exc

            archive = _zip_outputs(run_dir, include_object=capture_held_object)
            return (
                f"Finished on {device.upper()}. The ZIP contains the GLB, GIF preview, and available "
                "motion/diagnostics files. Uploaded inputs and intermediate run files "
                "are discarded after this job. SAM uses camera-aligned motion: "
                "camera movement is not removed and spine joints are approximated. "
                + (
                    "Main-person tracking was enabled: inspect subject selection; "
                    "extra detections may have been ignored. "
                    if track_person else ""
                )
                + (
                    "The object GLB and motion are approximate and camera-relative; "
                    "they are not attached or aligned to the character. "
                    "Inspect object_metadata.json and missing frames. "
                    if capture_held_object else ""
                )
                + (
                    "Quality report: diagnostics.json in the ZIP."
                    if (run_dir / "diagnostics.json").is_file()
                    else "Quality report is missing; re-check the downloaded run with m2mr diagnose."
                ),
                str(archive),
            )
    finally:
        for upload in uploads:
            if upload:
                uploaded_file = Path(upload)
                try:
                    uploaded_file.resolve().relative_to(GRADIO_TEMP_DIR.resolve())
                except ValueError:
                    continue
                uploaded_file.unlink(missing_ok=True)


with gr.Blocks(delete_cache=(86400, 86400), title="4onstudios LTD") as demo:
    gr.Markdown(
        """
# 4onstudios LTD
### Motion2MixamoRig open-source motion capture pipeline
Generate an animated Mixamo character (`.glb`) from a single-person video or still pose.
Character upload is optional: leave it empty to animate the Space's default
T-pose mannequin, or upload your own Mixamo-rigged binary FBX.
Without an owner-configured character, the default is the built-in blue low-poly doll.

**SAM 3D Body:** no SMPL-X download is needed. The Space owner must accept
[Meta's model terms and request access](https://huggingface.co/facebook/sam-3d-body-dinov3),
set an `HF_TOKEN` secret with model read access, and select NVIDIA GPU hardware.
Use is subject to the SAM license, detector-weight terms and your character/media rights.

**Experimental:** use a fixed, level camera with one person visible throughout.
This adapter smooths per-frame estimates; it does not remove camera movement,
or recover a gravity-aligned world trajectory. Optional native SAM finger animation
estimates curl and splay, not axial twist. It needs clearly visible hands and a
character with all 30 Mixamo finger bones (the Default Mannequin has them).
Occluded hands may still be guessed incorrectly; full inference is slower.

Uploaded video/image and Mixamo FBX files are used for this job only. The output
ZIP and object-selection previews are kept in temporary Space cache for up to
24 hours; run files and uploads are not retained in the attached Storage Bucket.
"""
    )
    with gr.Row():
        media_type = gr.Radio(
            ["Video", "Still image"], value="Video", label="Input type"
        )
        rig_upload = gr.File(
            label="Optional character (binary Mixamo FBX, T-pose)",
            file_types=[".fbx"],
            type="filepath",
        )
    with gr.Row():
        video_upload = gr.File(
            label="Single-person video",
            file_types=sorted(VIDEO_SUFFIXES),
            type="filepath",
        )
        image_upload = gr.File(
            label="Single-person still image",
            file_types=sorted(IMAGE_SUFFIXES),
            type="filepath",
            visible=False,
        )
    media_type.change(
        lambda value: (
            gr.update(visible=value == "Video"),
            gr.update(visible=value == "Still image"),
        ),
        inputs=media_type,
        outputs=[video_upload, image_upload],
        api_name=False,
    )
    detect_hands = gr.Checkbox(label="Animate fingers (native SAM, experimental)", value=False)
    track_person = gr.Checkbox(
        label="Track main person (ignore extra background detections, experimental)",
        value=False,
        info=(
            "Starts with the dominant largest detection, then follows box overlap. "
            "Stops if lost or ambiguous. Not appearance-based identity tracking; "
            "inspect results when people overlap."
        ),
    )
    hand_smooth = gr.Slider(
        minimum=0, maximum=0.3, value=0.08, step=0.01,
        label="Finger smoothing (seconds)",
    )
    with gr.Accordion("Capture a separate object (video only, experimental)", open=False):
        capture_held_object = gr.Checkbox(
            label="Capture the selected object", value=False,
            info=(
                "Requires separately configured SAM 2, TripoSR and MegaPose GPU runtimes. "
                "The result is not attached or aligned to the character."
            ),
        )
        with gr.Row():
            object_frame = gr.Number(value=0, precision=0, label="Frame where the object is visible (0-based)")
            object_size = gr.Number(
                value=None, minimum=0.01, label="Approximate longest object dimension (cm)",
            )
        load_object_frame = gr.Button("Load object frame")
        object_preview = gr.Image(label="Click two opposite corners around the object", interactive=False)
        object_state = gr.State()
        object_message = gr.Textbox(label="Object selection", interactive=False)
        load_object_frame.click(
            preview_object_frame, inputs=[video_upload, object_frame],
            outputs=[object_preview, object_state, object_message], api_name=False,
        )
        object_preview.select(
            select_object_point, inputs=[video_upload, object_frame, object_state],
            outputs=[object_preview, object_state, object_message], api_name=False,
        )
    run_button = gr.Button("Generate GLB", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    result = gr.File(label="Download output bundle", interactive=False)
    run_button.click(
        generate,
        inputs=[
            media_type, video_upload, image_upload, rig_upload,
            detect_hands, hand_smooth, track_person,
            capture_held_object, object_frame, object_state, object_size,
        ],
        outputs=[status, result],
        concurrency_limit=1,
        api_name=False,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
