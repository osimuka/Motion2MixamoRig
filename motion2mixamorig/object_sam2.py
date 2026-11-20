"""Isolated SAM 2 video segmentation worker for one selected object."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def mask_bounding_box(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Object mask must be two-dimensional.")
    y, x = np.nonzero(mask)
    if len(x) == 0:
        return None
    return int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1


def segment_video(
    frames: Path, frame_count: int, selected_frame: int, box: np.ndarray,
    checkpoint: Path, output: Path, seed_mask: Path,
) -> None:
    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    if not torch.cuda.is_available():
        raise RuntimeError("Object segmentation requires an NVIDIA CUDA GPU.")
    predictor = build_sam2_video_predictor(
        "configs/sam2.1/sam2.1_hiera_t.yaml", str(checkpoint), device="cuda",
    )
    boxes = np.full((frame_count, 4), np.nan, dtype=np.float32)
    masks_seen = np.zeros(frame_count, dtype=bool)

    with torch.inference_mode():
        state = predictor.init_state(
            video_path=str(frames), offload_video_to_cpu=True, offload_state_to_cpu=True,
        )
        if state["num_frames"] != frame_count:
            raise ValueError("SAM 2 decoded a different number of video frames.")
        initial = predictor.add_new_points_or_box(
            inference_state=state, frame_idx=selected_frame, obj_id=1,
            box=np.asarray(box, dtype=np.float32),
        )

        def save_mask(result) -> None:
            index, ids, logits = result
            if not 0 <= index < frame_count or 1 not in ids:
                raise ValueError("SAM 2 returned an invalid object or frame index.")
            mask = logits[list(ids).index(1), 0].detach().cpu().numpy() > 0
            if mask.shape != (state["video_height"], state["video_width"]):
                raise ValueError("SAM 2 returned an object mask with the wrong dimensions.")
            bbox = mask_bounding_box(mask)
            masks_seen[index] = True
            if bbox is not None and bbox[2] - bbox[0] >= 2 and bbox[3] - bbox[1] >= 2:
                boxes[index] = bbox
            if index == selected_frame:
                if bbox is None or not np.isfinite(boxes[index]).all():
                    raise ValueError("The selected object produced an empty mask. Select it again.")
                if not cv2.imwrite(str(seed_mask), mask.astype(np.uint8) * 255):
                    raise RuntimeError("Could not save the selected object mask.")

        save_mask(initial)
        for reverse in (False, True):
            for result in predictor.propagate_in_video(
                state, start_frame_idx=selected_frame, reverse=reverse,
            ):
                save_mask(result)
    if not masks_seen.all():
        missing = np.flatnonzero(~masks_seen)
        raise RuntimeError(f"SAM 2 did not process {len(missing)} source frames.")
    np.savez_compressed(output, boxes=boxes, visible=np.isfinite(boxes).all(axis=1))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, required=True)
    parser.add_argument("--selected-frame", type=int, required=True)
    parser.add_argument("--box", type=float, nargs=4, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed-mask", type=Path, required=True)
    args = parser.parse_args()
    segment_video(
        args.frames, args.frame_count, args.selected_frame, np.array(args.box),
        args.checkpoint, args.output, args.seed_mask,
    )


if __name__ == "__main__":
    main()
