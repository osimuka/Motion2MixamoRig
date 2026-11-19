## Motion2MixamoRig v0.1.5

Motion2MixamoRig transfers human motion from a monocular video—or a pose from a still image—onto an Adobe Mixamo-rigged 3D character.

### Highlights

- **Blender 4.2+ Extension**: pick a video or image and a Mixamo character in the Blender sidebar, run the pipeline in an external Python environment, follow its progress, and import the generated character directly into Blender
- New CLI options for automation: `--output-dir`, `--progress-jsonl`, and `--no-preview`
- README available in 18 languages

### Install the Blender Extension

Download `motion2mixamorig-0.1.5.zip` from the assets below, then in Blender 4.2+: **Edit → Preferences → Get Extensions → ∨ → Install from Disk…**. Setup and usage: [`blender_extension/README.md`](https://github.com/osimuka/Motion2MixamoRig/blob/main/blender_extension/README.md).

### Breaking change

- **Python 3.11+ is now required.** The `gvhmr` dependency does not support Python 3.10.

### Important notes

* Input currently supports one clearly visible, full-body person.
* The first run downloads approximately 5 GB of inference weights.
* The SMPL-X body model and Mixamo character must be obtained separately under their respective licenses.
* Motion extraction and retargeting are experimental.
* Some motions may exhibit root drift, orientation errors, foot sliding, or unstable rotations.
