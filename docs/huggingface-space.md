# 4onstudios LTD — Private Hugging Face Space

This 4onstudios LTD improvement fork of
[Motion2MixamoRig](https://github.com/MaxLiu22/Motion2MixamoRig) provides a
browser UI for one-person video or still-image input and a Mixamo character
FBX. It produces a ZIP containing `mixamo_character.glb` and the available
motion/diagnostics files. Input files and intermediate run data are temporary;
the ZIP is removed from the Space's Gradio cache after 24 hours.

## Licensing

Use this deployment only for non-commercial research or private
experimentation unless you have separately obtained all required commercial
rights. The current pipeline depends on GVHMR and SMPL-X, whose upstream terms
restrict use to non-commercial scientific research. A private Space does not
change those terms. YOLOv8 and other upstream components also have their own
licenses; see [`NOTICE.md`](../NOTICE.md) and each upstream license before use.

The SMPL-X model and Mixamo characters are user-supplied licensed assets. Do
not add them to the Space Git repository or publish them in a public dataset.

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
   it at `/data`. The app stores the licensed SMPL-X file and downloaded
   inference weights there. Allow several GB for the first-run model weights.
   If you mount the bucket elsewhere, set the Space variable
   `M2MR_SPACE_DATA_ROOT` to that mount path.
4. Put your licensed `SMPLX_NEUTRAL.npz` in the bucket at:

   ```text
   assets/body_models/smplx/SMPLX_NEUTRAL.npz
   ```

   The bucket is persistent across Space restarts. The app does not upload
   videos or Mixamo files into it.
5. In Space **Settings → Hardware**, choose a GPU for practical video
   processing. Start with a single Nvidia L4 or A10G and adjust if needed. The
   app selects CUDA automatically when available and falls back to CPU
   otherwise. A CPU run can be much slower. First use downloads roughly 5 GB of
   model weights into the bucket; later runs reuse them.
6. Wait for the Docker build to complete, then open the Space. Upload one
   single-person video or still image and a binary Mixamo FBX in T-pose, and
   select **Generate GLB**. Download the result ZIP before it expires.

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

## Local app smoke test

The Space uses the repository's Dockerfile and requires an NVIDIA GPU for
accelerated inference. For local checks, configure
`M2MR_SPACE_DATA_ROOT` to a directory containing the SMPL-X file, build the
Docker image, and open `http://localhost:7860`. Running without a GPU is
possible but inference will be slower.
