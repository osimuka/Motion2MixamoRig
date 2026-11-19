## Motion2MixamoRig v0.1.6

Motion2MixamoRig transfers human motion from a monocular video—or a pose from a still image—onto an Adobe Mixamo-rigged 3D character.

This release makes the Blender add-on usable without a terminal.

### Highlights

- **One-click runtime**: click **Install Runtime** in the add-on and it downloads Python and all dependencies by itself. No terminal, no Python install.
- **Choose SMPL-X Model…** right in the panel
- **First-run download progress**: size, speed, and time left while the models (~5.6 GB) download
- **Add-on UI in 18 languages**, the same set as the README
- A **⚙** next to the panel title hides the Setup box
- A step-by-step [User Guide](https://github.com/osimuka/Motion2MixamoRig/blob/main/docs/user-guide.md) for non-programmers

### Install the Blender add-on

1. Download `motion2mixamorig-0.1.6-macos-arm64.zip` from the assets below (do not unzip it). This release has a macOS (Apple Silicon) build only; Windows and Linux builds will follow once they are tested.
2. In Blender 4.2+: **Edit → Preferences → Get Extensions → ∨ → Install from Disk…**
3. Follow the [User Guide](https://github.com/osimuka/Motion2MixamoRig/blob/main/docs/user-guide.md).

Updating from an earlier version: remove the old add-on, quit Blender, install this zip, then click **Update Runtime**.

### Fixes

- The original-video panel of `compare.mp4` no longer plays at half speed for 60 fps (and other non-30 fps) clips.
- The fallback YOLO weights now download into `weights/yolo/`, not the working directory.

### Important notes

* Tested on macOS (Apple Silicon) only. Windows and Linux are untested.
* The first run downloads about 1.4 GB of runtime and 5.6 GB of models, and needs about 8 GB of free disk space.
* Input currently supports one clearly visible, full-body person.
* The SMPL-X body model and Mixamo character must be obtained separately under their respective licenses.
* Non-commercial use only, because of upstream model licenses (see `NOTICE.md`).
* Motion extraction and retargeting are experimental; some motions may show root drift, foot sliding, or unstable rotations.
