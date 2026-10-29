# NOTICE

This project's own code is released under the license in `LICENSE`. It builds
on the following upstream components, each governed by its own terms. None of
their gated assets are redistributed here; users obtain them directly from the
sources below.

## GVHMR

World-Grounded Human Motion Recovery via Gravity-View Coordinates
(Shen et al., SIGGRAPH Asia 2024). Installed as the `gvhmr` Python package.
GVHMR is released for **non-commercial research purposes only** — see its
license before any commercial use.
https://github.com/zju3dv/GVHMR

## SAM 3D Body (Space / `--set-extractor sam3d`)

SAM 3D Body is distributed by Meta under the custom SAM License, covering its
code and model materials. Model access requires the user to request approval
and accept the terms on Hugging Face. The adapter uses the MHR representation
and does not require SMPL-X or MANO. MHR's own repository code is Apache-2.0;
the MHR TorchScript asset bundled with the checkpoint is part of the licensed
SAM model materials. The DINOv3 backbone also has its upstream terms.

https://github.com/facebookresearch/sam-3d-body/blob/main/LICENSE
https://huggingface.co/facebook/sam-3d-body-dinov3
https://github.com/facebookresearch/MHR
https://github.com/facebookresearch/dinov3

The SAM adapter uses TorchVision Faster R-CNN ResNet50 FPN V2 COCO weights for
person detection. TorchVision code uses a BSD-style license; consult upstream
for pretrained-weight and dataset terms. It does not use Ultralytics YOLO.

https://pytorch.org/vision/stable/models/generated/torchvision.models.detection.fasterrcnn_resnet50_fpn_v2.html

## SMPL-X body model

The SMPL-X body model is registration-gated and licensed by the Max Planck
Institute for Intelligent Systems for non-commercial scientific research.
Users must register and download `SMPLX_NEUTRAL.npz` themselves.
https://smpl-x.is.tue.mpg.de/

## Adobe Mixamo characters

Mixamo characters (Y Bot and others) are provided by Adobe under the Mixamo
terms of use. Download them with your own Adobe account; the FBX files are
never committed to this repository.
https://www.mixamo.com/

## Inference weights

The extraction step auto-downloads model checkpoints (GVHMR, HMR2, RTMPose /
ViTPose 2D pose, YOLOv8 person detector) into `weights/`. Note that YOLOv8
(Ultralytics) is licensed under **AGPL-3.0**, and the pose/HMR checkpoints
carry their respective research licenses.

## Summary

The GVHMR end-to-end path is effectively restricted to non-commercial /
research use by its upstream components, regardless of this repository's own
code license. The SAM Space path does not use GVHMR, SMPL-X, MANO or YOLO, but
is subject to its own model, backbone, detector and character/media terms.
Private hosting is not a substitute for obtaining the applicable rights.
