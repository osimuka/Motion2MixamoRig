# 4onstudios LTD — Private Hugging Face Space

This 4onstudios LTD improvement fork of
[Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) provides a
browser UI for one-person video or still-image input and an optional Mixamo
character FBX. Without a character upload it uses the Space's **Default
Mannequin**. The owner's blue model is included in the repository and can be
selected with `M2MR_DEFAULT_RIG`; without that configuration the app
uses an original, built-in blue low-poly mannequin with a Mixamo-compatible
skeleton and T-pose bind pose.
No Adobe character is bundled or downloaded. Uploaded characters take
precedence; an invalid upload reports an error instead of silently switching
to the mannequin. It produces a ZIP containing `mixamo_character.glb` and the available
motion/diagnostics files. The Space uses **SAM 3D Body**, not GVHMR: no SMPL-X
or MANO file is required. Input files and intermediate run data are temporary;
the ZIP is eligible for Gradio cache cleanup after 24 hours.

## Verified deployment

The operator confirmed successful video generation on the private Space
[`osimuka/4onstudios-motion`](https://huggingface.co/spaces/osimuka/4onstudios-motion)
with an NVIDIA L4 on **2026-10-29**. The deployment uses a one-hour inactivity
sleep timeout. This confirms the generation path worked for that run, not
animation quality, a measured memory budget, or compatibility with every input.

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

### Create the `HF_TOKEN` secret

1. Sign in to the same Hugging Face account that received SAM model approval.
2. Open [New access token](https://huggingface.co/settings/tokens/new).
   Create a **fine-grained** token named, for example, `4onstudios-motion`,
   with read access to `facebook/sam-3d-body-dinov3`. Enable permission to read
   public gated repositories you can access if shown in the token form.
   No write permission is needed for model downloads.
3. Copy the token, then open the Space's **Settings → Variables and secrets**.
4. Choose **New secret**, not a public variable. Set the name to `HF_TOKEN`
   and paste the token as its value. Save and restart the Space if necessary.

Do not paste tokens into chat, screenshots, source files, or Git commits.
Model approval and token permission are separate: creating a token does not
bypass the model's access gate.

### Default Mannequin

Set the Space variable `M2MR_DEFAULT_RIG` to an absolute path to a licensed,
binary Mixamo-compatible T-pose FBX inside the container. Uploaded characters
still take precedence. A missing or invalid configured default reports an
error rather than silently switching characters.

The Dockerfile copies `private_assets/` into `/app/private_assets/`.
The fork owner's [Default Mannequin](../private_assets/default_mannequin.fbx)
is included on GitHub at the owner's request. Set the Space variable
`M2MR_DEFAULT_RIG=/app/private_assets/default_mannequin.fbx` to use it.
Despite the historical directory name, this specific model is tracked;
other files under `private_assets/` remain git-ignored.
Do not add other third-party models without permission to redistribute them.

The supplied blue character was converted locally from GLB, preserving its
body mesh and diffuse-sampled colors, remapping the skeleton to Mixamo names, and
removing the racket so the default stays a general-purpose mannequin.
The current FBX/GLB pipeline samples the diffuse texture into face/vertex
colors; it does not embed the original texture or preserve full source PBR
materials such as normal, roughness, transmission, and clearcoat maps.
The source pack and its other animations are not deployed.
The included asset retains its applicable asset rights; inclusion does not
grant the project's code license to the model. It is separate from the
fork's procedural mannequin.

When adapting another GLB, convert locally using Blender and verify the
T-pose, bone names, scale, normalized skin weights, and bind-pose error before
deployment. The uploader itself still accepts only binary FBX, not GLB.

### First generation and downloads

1. Wait until the Space is **Running** on the selected NVIDIA GPU, not merely
   showing that GPU as requested. The UI can respond before GPU activation.
2. Select **Video** and upload a short, licensed 3–5 second clip with one
   person visible head to toe and a fixed, level camera. Supported video
   extensions are `.avi`, `.mkv`, `.mov`, `.mp4`, and `.webm`.
   **Still image** accepts `.bmp`, `.jpeg`, `.jpg`, `.png`, and `.webp`.
3. Leave **Optional character** empty to animate the Default Mannequin,
   or supply a binary Mixamo FBX in T-pose.
4. Select **Generate GLB**. The first generation downloads model weights and
   takes longer than subsequent runs; keep the page open until it finishes.
5. Download and extract the output ZIP:
   - `mixamo_character.glb`: skinned character and animation; import through
     Blender's **File → Import → glTF 2.0**, or copy it into a Godot 4 project.
     See the [Godot guide](godot.md) for playback and vertex-color materials.
   - `mixamo_rotations.npz`: per-bone quaternion rotations, wxyz convention.
   - `skeleton_motion.npz`: extracted 24-joint motion and timing.
   - `diagnostics.json`: automatic self-intersection quality report, when
     successfully generated. Frames are numbered from zero.

Inspect the animation and quality report before using the result in a game or
animation. An intersection finding does not fail generation. If the report is
missing, the status explicitly says so; after extracting the ZIP into a run
directory, use `m2mr diagnose <run> --visualize` in a configured local environment
to regenerate it and create an HTML view.

### Hardware costs and sleep

GPU Spaces incur charges while running, even without generation requests.
Check the current price in Hugging Face's hardware selector before upgrading.
Configure an inactivity sleep timeout (the verified deployment uses one hour),
and pause the Space manually when finished if you want to stop compute sooner.
Waking a sleeping Space can take time; confirm it is running before retrying.

### Native SAM fingers

Enable **Animate fingers (native SAM, experimental)** before generating.
It is off by default to preserve body-only behavior and processing cost.
**Finger smoothing** controls the rotation smoothing window in seconds
(default 0.08; zero disables smoothing).

- Uses SAM's native `full` inference with body and hand decoders and the same
  approved checkpoint. No HaMeR, MANO, SMPL-X, or new hand-model download.
- Needs a character with all 30 Mixamo finger bones. The included Default
  Mannequin supports this; the procedural low-poly fallback has no fingers
  and produces an explicit error if native fingers are requested.
- Saves `(frames, 2, 21, 3)` Y-up hand landmarks in the skeleton NPZ under
  `sam_hand_keypoints` (left, then right; wrist, then thumb/index/middle/ring/
  pinky, each base-to-tip). Still-image landmarks repeat with the 32-frame hold.
- Maps palm-relative segment directions into rig-relative finger rotations.
  This estimates curl and splay, not axial twist or independent wrist twist.
  Degenerate hand geometry is flagged in logs; short gaps can be smoothed,
  while unfilled gaps remain at rest. No usable hands is a generation error.
- The landmark output has no visibility confidence. Occluded or tiny hands
  may still produce incorrect poses. Use close, sharp footage and inspect
  the animation. Full inference is slower; GPU memory/speed are not benchmarked.
- Reusing a skeleton containing these landmarks also reuses its fingers
  without inference. A body-only SAM skeleton cannot acquire fingers from the
  GVHMR cache: regenerate the source with native hands enabled instead.

The CLI equivalent inside the SAM runtime is:

```bash
python -m motion2mixamorig.cli run --set-extractor sam3d --device cuda \
  --video clip.mp4 --rig character.fbx --detect-hands --hand-smooth 0.08
```

Synthetic tests cover landmark ordering, left/right chains, camera-rotation
invariance, rotation/GLB export and skeleton reuse. Real native-hand inference
and visual finger quality still need validation on the GPU Space.

### Troubleshooting

| Symptom | Action |
| --- | --- |
| Upload error lists `..mp4` or other double-dot extensions | Deploy the corrected upload filters, wait for the rebuild, refresh the page, and upload again. A valid `.mp4` does not need renaming. |
| `PACKAGE_MISSING` / `No module named 'termcolor'` | Update to the current Dockerfile and `requirements-space.txt`, then rebuild. `termcolor` is a DINOv3 dependency, not a missing Motion2MixamoRig installation. |
| Model access denied | Confirm SAM approval for the token's account, gated-model read permission, and the exact secret name `HF_TOKEN`; restart after changing it. |
| CUDA unavailable | Confirm the active hardware is NVIDIA GPU, not CPU Basic with a GPU upgrade still pending. |
| Missing or multiple people | Use a clip with exactly one clearly visible person throughout; no cuts or people entering the background. |
| Generic generation failure | Inspect the Space runtime logs for the traceback. Remove credentials and private media paths before sharing logs. |

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
  Body-only inference leaves individual Mixamo fingers at rest unless native
  hands are enabled; see [Native SAM fingers](#native-sam-fingers).
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
the generated ZIP is temporarily held in Gradio's cache and becomes eligible
for cleanup after 24 hours. Cleanup is periodic, not a guaranteed deletion at
exactly 24 hours. Download promptly. Hugging Face controls the Space infrastructure, so do not
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

The Docker build runs the SAM and mannequin regression suites and the DINOv3
Hub import check, including synthetic native-hand tests.
Unit tests use synthetic predictions; they are not evidence of model quality.
The operator separately confirmed successful generation on the approved L4
deployment. Animation quality, peak memory usage, and processing speed still
need evaluation on representative clips.
