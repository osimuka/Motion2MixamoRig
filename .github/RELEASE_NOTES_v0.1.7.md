## Motion2MixamoRig v0.1.7

Motion2MixamoRig transfers human motion from a monocular video—or a pose from a still image—onto an Adobe Mixamo-rigged 3D character.

This release adds support for Mixamo characters that were edited and re-exported in Cinema 4D.

### Highlights

- **Cinema 4D-exported Mixamo FBX support**: use a character you opened and re-exported in Cinema 4D directly as the Mixamo rig, in the Blender add-on or with `--rig`. Details: [`docs/c4d-fbx-support.md`](https://github.com/osimuka/Motion2MixamoRig/blob/main/docs/c4d-fbx-support.md).
- **Fixed characters falling apart in the GLB**: each mesh piece now keeps its own skin bind matrices.

Both were contributed by [@yng47](https://github.com/yng47). Thank you!
- **User guide in 18 languages**, matching the README.

### What you will notice

- In Blender the character now imports as one mesh object per piece (Y Bot: `skinned_mesh_0_Alpha_Surface` and `skinned_mesh_1_Alpha_Joints`) instead of a single `skinned_mesh`. The armature is still named `mixamo`, and the animation is unchanged.
- Original Mixamo characters produce the same motion as v0.1.6.

### Install the Blender add-on

1. Download `motion2mixamorig-0.1.7-macos-arm64.zip` from the assets below (do not unzip it). This release has a macOS (Apple Silicon) build only.
2. In Blender 4.2+: **Edit → Preferences → Get Extensions → ∨ → Install from Disk…**
3. Follow the [User Guide](https://github.com/osimuka/Motion2MixamoRig/blob/main/docs/user-guide.md).

Updating from v0.1.6: remove the old add-on, quit Blender, install this zip, then click **Update Runtime** (the C4D support lives in the runtime code, so this step is required). Downloaded models are kept.

### Important notes

* Tested on macOS (Apple Silicon) only. Windows and Linux are untested.
* Input currently supports one clearly visible, full-body person.
* The SMPL-X body model and Mixamo character must be obtained separately under their respective licenses.
* Non-commercial use only, because of upstream model licenses (see `NOTICE.md`).
