# Maintained Fork Notice

`vnccs-utils-maintained` is a maintained fork of
[AHEKOT/ComfyUI_VNCCS_Utils](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils).
The original project and its bundled assets remain subject to their existing
copyright notices and licenses. The original VNCCS code and this fork's
MIT-licensed contributions retain the upstream MIT terms. Bundled and derived
third-party components remain under the licenses identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Compatibility

This fork is a drop-in replacement for the upstream `vnccs-utils` package. It
intentionally keeps the original ComfyUI node identifiers so existing
workflows continue to load without node migration.

Do not install the fork and upstream package in the same ComfyUI instance.
They register the same node identifiers and would conflict. The GitHub
repository, Python project, Registry package, display name, and release
versions are distinct so users can tell the maintained fork from upstream.

## Material Differences

The maintained fork adds a direct Pose Studio OpenPose output and replaces the
upstream Pose Studio execution path with a single-pass resolver shared by both
Pose Studio nodes. It also corrects:

* alignment between captures, prompts, lighting, dimensions, and OpenPose data;
* MakeHuman skin-weight canonicalization and morph-factor parity;
* OpenPose validation, face landmarks, hand data, and grid translation;
* Mixamo animation sampling limits and endpoint preservation;
* browser/server capture, pose, lighting, prompt, and pixel limits;
* stale capture-cache handling and synchronization failures; and
* unsolicited Pose Studio repository refresh and passive CDN access.

These changes are covered by focused regression tests and live ComfyUI
validation.

## Related ComfyUI Nodes

Several maintained nodes overlap with parts of Pose Studio, but none is a
drop-in replacement for the complete VNCCS Utils package:

| Project | Primary scope | Relationship |
| --- | --- | --- |
| [VNCCS Utils](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils) | Original Pose Studio, UniCanvas, and VNCCS utilities | Direct upstream; install either upstream or this maintained fork |
| [ComfyUI OpenPose Studio](https://github.com/andreszs/ComfyUI-OpenPose-Studio) | 2D OpenPose keypoint editing and pose gallery | Overlaps OpenPose authoring, not the parametric MakeHuman renderer or broader VNCCS utility set |
| [ComfyUI 3D Pose Editor](https://github.com/hinablue/ComfyUI_3dPoseEditor) | Interactive 3D skeleton posing | Overlaps 3D posing, not a workflow-compatible VNCCS replacement |
| [Yedp Action Director](https://github.com/yedp123/ComfyUI-Yedp-Action-Director) | Multi-character 3D action direction and Mixamo workflows | Broader action-direction alternative with meaningful pose and Mixamo overlap, but different nodes and workflows |
| [ComfyUI VRM Pose Editor](https://github.com/ketle-man/comfyui-vrm-pose-editor) | Posing user-supplied VRM, GLB, and GLTF models | Uses external avatar rigs rather than VNCCS Pose Studio's generated mannequin |

This fork is published as maintenance of the VNCCS implementation, not as a
claim that 2D or 3D pose editing is unique to it.
