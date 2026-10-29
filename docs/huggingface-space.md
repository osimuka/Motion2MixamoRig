# 4onstudios LTD — Private Hugging Face Space

This 4onstudios LTD improvement fork of
[Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) provides a
browser UI for one-person video or still-image input and an optional Mixamo
character FBX. Without a character upload it uses an original, built-in blue
low-poly mannequin with a Mixamo-compatible skeleton and T-pose bind pose.
No Adobe character is bundled or downloaded. Uploaded characters take
precedence; an invalid upload reports an error instead of silently switching
to the mannequin. It produces a ZIP containing `mixamo_character.glb` and the available
motion/diagnostics files. The Space uses **SAM 3D Body**, not GVHMR: no SMPL-X
or MANO file is required. Input files and intermediate run data are temporary;
the ZIP is eligible for Gradio cache cleanup after 24 hours.

## Licensing

SAM 3D Body has Meta's [custom SAM license](https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE),
not an MIT/Apache license. Request model access and accept those terms yourself.
Its MHR representation avoids the separate SMPL-X license. The adapter uses
TorchVision's Faster R-CNN person detector, not YOLO; TorchVision's code license
does not automatically settle the training-data or weight-use terms. Review
all applicable upstream terms and rights to your source media and Mixamo
character before commercial use. See [`NOTICE.md`](../NOTICE.md).

The existing CLI/Blender GVHMR path remains available, with its original
research-only restrictions. Switching the Space does not relicense GVHMR.

## Create the Space

1. Create a Hugging Face **Docker Space** and set its visibility to **Private**
   (not Protected). Invite only the collaborators who should be able to access
   the code and running app. Private Spaces require a Hugging Face plan that
   supports private Spaces.
2. Push this repository to the Space repository. If this checkout already has
   a GitHub `origin`, add the Space as another remote:

   ```bash
   git remote add hf https://huggingface.co/spaces/<account>/<space-name>
   git push hf main
   ```

3. In Space **Settings**, attach a private Hugging Face Storage Bucket and mount
   it at `/data`. The app stores downloaded SAM and detector weights there.
   Allow several GB for the model weights and sufficient space for download
   temporaries.
   If you mount the bucket elsewhere, set the Space variable
   `M2MR_SPACE_DATA_ROOT` to that mount path.
4. Request access at
   [facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3).
   Once approved, add **HF_TOKEN** as a Space **secret**, using a token with
   permission to read that gated model. A token on your laptop is not
   automatically available inside the Space. Never put it in code, Git or chat.
5. In Space **Settings → Hardware**, choose an NVIDIA GPU. Consider a single
   L4 or A10G as a starting point, but memory requirements depend on the
   checkpoint and inputs and have not yet been measured for this adapter.
   **CPU/MPS inference is not supported** by the supplied SAM estimator.
   No paid hardware is selected automatically. First use downloads the model
   into the bucket; later runs reuse it.
6. Wait for the Docker build to complete, then open the Space. Upload one
   single-person video or still image. Optionally upload a binary Mixamo FBX
   in T-pose; leave it empty to use the default mannequin. Then
   select **Generate GLB**. Download the result ZIP before it expires.

## Motion limitations

This is an experimental adapter around an image-based estimator, not a
replacement for GVHMR's world-motion tracking:

- Use a **fixed, level camera**, no cuts, one person, head to toe throughout.
  Detection runs on every frame; missing or multiple people stop the job rather
  than dropping frames, freezing a pose or switching identities.
- SAM's metre-valued camera coordinates plus `pred_cam_t` are rotated to Y-up.
  Initial pelvis X/Z are centered and the lowest initial ankle/toe is placed at
  Y=0. Camera movement and tilt are **not** compensated.
- Spine/collar landmarks are approximated from hip/shoulder/neck landmarks.
  The adapter uses body-only inference; individual Mixamo fingers stay at rest.
  `--detect-hands` is rejected for `sam3d`.
- A 40 ms Gaussian filter reduces jitter offline. No foot-contact probabilities
  are fabricated; GVHMR contact-based foot locking is unavailable.
- Every decoded video frame is retained and output FPS follows source duration.
  A still is exported as a 32-frame hold at 30 FPS.
- Inspect the animation and `diagnostics.json` before use. If diagnostics fail,
  the download status says the quality report is missing.

## Privacy and data retention

Private Spaces limit both the source repository and running app to the owner
and invited collaborators. This is separate from the bucket's visibility:
keep the bucket private as well.

Each run's uploaded files and generated working directory are held only in
temporary container storage. The app removes the uploaded files after the run;
the generated ZIP is temporarily held in Gradio's cache for up to 24 hours so
it can be downloaded. Hugging Face controls the Space infrastructure, so do not
upload sensitive media unless its hosting and retention terms are acceptable
for your use.

## Local setup and validation

Build the repository Dockerfile, then run with an NVIDIA GPU and your
credentials supplied through the environment:

```bash
docker build -t m2mr-sam3d .
docker run --rm --gpus all -p 7860:7860 \
  -e HF_TOKEN -v m2mr-models:/data m2mr-sam3d
```

Open `http://localhost:7860`. The UI can start without CUDA, but generation will
explicitly report that GPU hardware is required.

The image runs the application source directly from `/app` and installs
[`requirements-space.txt`](../requirements-space.txt), deliberately avoiding the
default GVHMR dependency stack. SAM's source is pinned in the Dockerfile and
exposed on `PYTHONPATH` because upstream does not supply an installable Python
package.

The build also imports DINOv3's Torch Hub entry points without downloading
weights, catching dependencies such as `termcolor` that load only when the
backbone is initialized. `PACKAGE_MISSING` errors name the missing dependency;
they do not necessarily mean Motion2MixamoRig itself is absent.

The same environment supports the CLI:

```bash
python -m motion2mixamorig.cli run --set-extractor sam3d --device cuda --video clip.mp4 --rig character.fbx
python -m unittest scripts.test_sam3d scripts.test_mannequin
```

SAM checkpoint access, CUDA inference, animation quality and memory usage must
be validated on an approved GPU deployment. Unit tests use synthetic predictions;
they are not evidence of model quality.
