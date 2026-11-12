## Motion2MixamoRig v0.1.9

### Highlights

- Preview-enabled runs now save `mixamo_character.gif` directly in the output folder and list it in `run.json`.
- Video GIFs show the character animation; still-image GIFs show the turntable comparison. They loop silently, fit within 480 x 480 pixels, and preserve the clip duration at up to 15 fps.
- Floor alignment now covers stills and clips without foot-contact data. A constant height offset grounds the lowest sole without flattening vertical movement.
- Grounded previews use floor height zero, matching the exported animation.

### Installation

This is a source-only release. No platform-specific Blender extension installers are attached because a full Blender add-on smoke test was not performed for this version. Use the source installation instructions in the README. Pillow is now an explicit dependency, so update the Python environment when updating the source.

`--no-preview` skips GIF generation, including jobs using that option through the Blender extension. To generate a GIF, run the CLI with previews enabled.

### Validation And Limitations

- Grounding, GIF timing/color/loop, and existing no-preview pipeline export regression tests passed.
- Synthetic video and still-image pipeline checks generated GIFs, recorded them in `run.json`, and wrote `diagnostics.json`.
- Full model inference and Blender import/add-on smoke tests were not performed for this release.
- Without detected contacts, the fallback assumes the lowest sole should touch the floor. Fully airborne clips may need a manual height adjustment.

See the [changelog](https://github.com/osimuka/Motion2MixamoRig/blob/v0.1.9/CHANGELOG.md) for details.