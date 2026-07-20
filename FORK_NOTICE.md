# Lineage and Compatibility Notice

`jakkanna` is an independently governed downstream distribution derived from
[AHEKOT/ComfyUI_VNCCS_Utils](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils).
Its initial code baseline is upstream version `0.5.3` at commit `1908ddf`.
Jakkanna selectively incorporates later upstream work; its governance,
roadmap, and version numbers are independent.

Jakkanna is not affiliated with or endorsed by the upstream VNCCS project.
The original project and its bundled assets remain subject to their existing
copyright notices and licenses. Original VNCCS code and MIT-licensed Jakkanna
contributions retain the upstream MIT terms. Bundled and derived third-party
components remain under the licenses identified in
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Compatibility

Jakkanna is a drop-in replacement for the upstream `vnccs-utils` package. It
intentionally keeps the original `VNCCS_*` ComfyUI node identifiers,
`/vnccs/` server routes, serialized fields, and browser compatibility hooks so
existing workflows continue to load without migration.

Do not install Jakkanna and the upstream package in the same ComfyUI instance.
They register the same node identifiers and would conflict. The GitHub
repository, Python project, Registry package, display name, and release
versions are distinct.

The earlier `vnccs-utils-maintained` Registry package and its `0.6.0` release
are a superseded pre-brand publication of this work. Jakkanna starts at
`1.0.0`; no relationship between Jakkanna and upstream version numbers should
be inferred.

## Material Differences

Jakkanna adds a direct Pose Studio OpenPose output and replaces the
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
| [VNCCS Utils](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils) | Original Pose Studio, UniCanvas, and VNCCS utilities | Source project; install either upstream or Jakkanna |
| [ComfyUI OpenPose Studio](https://github.com/andreszs/ComfyUI-OpenPose-Studio) | 2D OpenPose keypoint editing and pose gallery | Overlaps OpenPose authoring, not the parametric MakeHuman renderer or broader VNCCS utility set |
| [ComfyUI 3D Pose Editor](https://github.com/hinablue/ComfyUI_3dPoseEditor) | Interactive 3D skeleton posing | Overlaps 3D posing, not a workflow-compatible VNCCS replacement |
| [Yedp Action Director](https://github.com/yedp123/ComfyUI-Yedp-Action-Director) | Multi-character 3D action direction and Mixamo workflows | Broader action-direction alternative with meaningful pose and Mixamo overlap, but different nodes and workflows |
| [ComfyUI VRM Pose Editor](https://github.com/ketle-man/comfyui-vrm-pose-editor) | Posing user-supplied VRM, GLB, and GLTF models | Uses external avatar rigs rather than VNCCS Pose Studio's generated mannequin |

Jakkanna is published as an independent evolution of the VNCCS implementation,
not as a claim that 2D or 3D pose editing is unique to it.
