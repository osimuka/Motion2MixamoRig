## Motion2MixamoRig v0.1.10

### Fixed

- Hugging Face Space jobs now generate `mixamo_character.gif` in the output folder and include it at the root of the downloadable result ZIP alongside the GLB and available motion/diagnostics files.
- Video inputs produce a character-animation GIF; still inputs produce the turntable-comparison GIF. Existing downloads are unchanged; generate a new result to include the preview.
- The Space completion message and documentation now describe the included GIF. Rendering previews adds processing time to each job.

### Validation

- All 14 local mannequin and Space regression tests passed, including GIF encoding and ZIP inclusion across all four character-selection cases.
- The Hugging Face Docker build passed all 37 tests after deploying this integration.
- Full model inference and a Blender add-on smoke test were not performed for this update.

### Installation

This is a source-only release with no platform-specific Blender extension installers attached. CLI and Blender jobs using `--no-preview` still skip GIF generation; the Space now enables previews automatically.

Full changes: [v0.1.9...v0.1.10](https://github.com/osimuka/Motion2MixamoRig/compare/v0.1.9...v0.1.10).