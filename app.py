"""Private Hugging Face Space UI for the 4onstudios LTD improvement fork."""

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
from motion2mixamorig.pipeline import run_pipeline
from motion2mixamorig.progress import ProgressWriter

logging.basicConfig(level=logging.INFO)
LOGGER = logging.getLogger("motion2mixamorig.space")

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
BUNDLE_FILES = (
    "mixamo_character.glb",
    "mixamo_rotations.npz",
    "skeleton_motion.npz",
    "diagnostics.json",
)


class SpaceProgressWriter(ProgressWriter):
    def __init__(self, progress: gr.Progress) -> None:
        super().__init__()
        self.progress = progress

    def emit(self, stage: str, progress: float, message: str) -> dict:
        event = super().emit(stage, progress, message)
        self.progress(event["progress"], desc=event["message"])
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


def _zip_outputs(run_dir: Path) -> Path:
    bundle_dir = GRADIO_TEMP_DIR / "downloads"
    bundle_dir.mkdir(parents=True, exist_ok=True)
    archive = bundle_dir / f"motion2mixamorig-{uuid.uuid4().hex}.zip"
    included = [run_dir / name for name in BUNDLE_FILES if (run_dir / name).is_file()]
    if not (run_dir / "mixamo_character.glb").is_file():
        raise RuntimeError("The pipeline completed without producing mixamo_character.glb.")
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for file in included:
            bundle.write(file, arcname=file.name)
    return archive


def generate(
    media_type: str,
    video_upload: str | None,
    image_upload: str | None,
    rig_upload: str | None,
    progress: gr.Progress = gr.Progress(),
) -> tuple[str, str]:
    uploads = (video_upload, image_upload, rig_upload)
    try:
        if not paths.SMPLX_NEUTRAL.is_file():
            raise gr.Error(
                "SMPL-X was not found. Attach the private Storage Bucket at "
                f"{SPACE_DATA_ROOT} and place the model at "
                "assets/body_models/smplx/SMPLX_NEUTRAL.npz."
            )
        _validate_upload(paths.SMPLX_NEUTRAL, checks.check_smplx, "SMPL-X model")

        with tempfile.TemporaryDirectory(prefix="m2mr-space-") as temp:
            job_dir = Path(temp)
            input_dir = job_dir / "input"
            input_dir.mkdir()
            rig = _copy_upload(rig_upload, input_dir, {".fbx"}, "character")
            _validate_upload(rig, checks.check_rig, "Mixamo character")

            if media_type == "Video":
                source = _copy_upload(video_upload, input_dir, VIDEO_SUFFIXES, "source")
                _validate_upload(source, checks.check_video, "Video")
                video, image = source, None
            elif media_type == "Still image":
                source = _copy_upload(image_upload, input_dir, IMAGE_SUFFIXES, "source")
                _validate_upload(source, checks.check_image, "Image")
                video, image = None, source
            else:
                raise gr.Error("Choose either Video or Still image.")

            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            run_dir = job_dir / "output"
            try:
                run_pipeline(
                    video,
                    rig,
                    image=image,
                    device=device,
                    verbose=False,
                    output_dir=run_dir,
                    progress=SpaceProgressWriter(progress),
                    no_preview=True,
                )
            except Exception as exc:
                LOGGER.exception("Motion2MixamoRig generation failed")
                code, message = classify_error(exc)
                raise gr.Error(f"Generation failed ({code}): {message}") from exc

            archive = _zip_outputs(run_dir)
            return (
                f"Finished on {device.upper()}. The ZIP contains the GLB and available "
                "motion/diagnostics files. Uploaded inputs and intermediate run files "
                "are discarded after this job.",
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
### Motion2MixamoRig improvement fork
Generate an animated Mixamo character (`.glb`) from a single-person video or still pose.

**Private use only:** the current GVHMR and SMPL-X components are restricted to
non-commercial research by their upstream terms. A private Space does not grant
commercial-use rights. Read the upstream terms in
[NOTICE.md](https://github.com/MaxLiu22/Motion2MixamoRig/blob/main/NOTICE.md).

Uploaded video/image and Mixamo FBX files are used for this job only. The output
ZIP is kept in temporary Space cache for up to 24 hours; run files and uploads
are not retained in the attached Storage Bucket.
"""
    )
    with gr.Row():
        media_type = gr.Radio(
            ["Video", "Still image"], value="Video", label="Input type"
        )
        rig_upload = gr.File(
            label="Mixamo character (binary FBX, T-pose)",
            file_types=[".fbx"],
            type="filepath",
        )
    with gr.Row():
        video_upload = gr.File(
            label="Single-person video",
            file_types=[f".{suffix}" for suffix in sorted(VIDEO_SUFFIXES)],
            type="filepath",
        )
        image_upload = gr.File(
            label="Single-person still image",
            file_types=[f".{suffix}" for suffix in sorted(IMAGE_SUFFIXES)],
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
    run_button = gr.Button("Generate GLB", variant="primary")
    status = gr.Textbox(label="Status", interactive=False)
    result = gr.File(label="Download output bundle", interactive=False)
    run_button.click(
        generate,
        inputs=[media_type, video_upload, image_upload, rig_upload],
        outputs=[status, result],
        concurrency_limit=1,
        api_name=False,
    )

demo.queue(default_concurrency_limit=1)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
