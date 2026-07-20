# Jakkanna

**Sculpting control for the AI age.**

Jakkanna is an independently governed ComfyUI custom-node suite for 3D posing,
OpenPose control, canvas composition, camera direction, regional detailing,
and model selection. It is derived from
[AHEKOT/ComfyUI_VNCCS_Utils](https://github.com/AHEKOT/ComfyUI_VNCCS_Utils)
and preserves its established workflow identifiers while following an
independent release and development path.

The name honours the legendary master sculptor popularly known as Jakanachari
or Jakkanna. It expresses the project's aim: precise control of form, movement,
framing, and composition.

> [!IMPORTANT]
> Install Jakkanna as a replacement for the upstream `vnccs-utils` package.
> Do not install both packages in the same ComfyUI instance. Jakkanna retains
> the original `VNCCS_*` node identifiers and `/vnccs/` routes so existing
> workflows continue to load.

## Why Jakkanna Exists

Jakkanna begins at version `1.0.0` with a correctness baseline that:

* Pose Studio and Pose Studio + OpenPose resolve one effective execution
  state and render from it exactly once.
* Captured images, lighting prompts, and OpenPose keypoints remain aligned
  across ordinary execution, frontend synchronization, cache fallback, and
  SAM-driven pose application.
* Skin weights are canonicalized, normalized, and limited to four influences
  per vertex.
* Browser and Python MakeHuman morph factors agree for body proportions and
  ancestry controls.
* Pose, capture, lighting, prompt, and total-pixel limits are consistent
  across browser and server boundaries.
* OpenPose validates frame structure and dimensions, exports real face
  landmarks, and translates people correctly in grid output.
* Mixamo sampling is hard-capped without dropping the animation endpoint.
* Repository refresh is explicitly user initiated; opening Pose Studio no
  longer starts an external repository refresh.

See [FORK_NOTICE.md](FORK_NOTICE.md) for lineage, compatibility, and a
comparison with related ComfyUI pose nodes.

## Who Jakkanna Is For

Use Jakkanna when a ComfyUI workflow needs an accurate production control
source. It is particularly useful for:

* existing VNCCS workflows that need the corrections without node migration;
* workflows that require rendered frames and body, hand, and face OpenPose
  data to remain exactly aligned;
* batch, grid, Mixamo, SAM-driven, pose-library, and MakeHuman morph workflows;
  and
* workflow authors who need deterministic output when sharing graphs with
  other users.

If you only need lightweight 2D keypoint editing and do not use VNCCS
workflows, a dedicated OpenPose editor may be a smaller fit.

<table>
<tr>
<td width="33%" align="center">
<strong>Upstream VNCCS Community</strong><br>
Share results, ask questions, and follow VNCCS updates.<br><br>
<a href="https://discord.com/invite/9Dacp4wvQw" target="_blank">Join the upstream Discord</a>
</td>
<td width="33%" align="center">
<strong>Support Jakkanna</strong><br>
Support ongoing compatibility work, testing, fixes, and releases.<br><br>
<a href="https://www.buymeacoffee.com/teenu" target="_blank">Support Jakkanna</a>
</td>
<td width="33%" align="center">
<strong>Support Original VNCCS</strong><br>
Support MiuProject's original VNCCS development separately.<br><br>
<a href="https://www.buymeacoffee.com/MIUProject" target="_blank">Support the original project</a>
</td>
</tr>
</table>

---

## Jakkanna Canvas

<p align="center">
  <img src="images/uni-canvas-logo.png" alt="Jakkanna Canvas" width="360">
</p>

**Jakkanna Canvas** is an integrated infinite-canvas image generation and editing workspace inside ComfyUI. It is designed for freeform creative work: generate anywhere, edit any region, build results across layers, and keep refining without being locked to a single fixed image boundary.

### Key Features

*   **Infinite Canvas Workflow**: Work beyond a single image frame and place generations wherever the composition needs them.
*   **Layer-Based Editing**: Build images from separate raster and mask layers with visibility, opacity, selection, movement, and compositing controls.
*   **Generation Anywhere**: Use a selected region as the generation target for new images, image edits, inpaint, outpaint, and full-area transformations.
*   **Mask and Object Tools**: Paint masks, refine selections, and use SAM-powered object selection to isolate or remove parts of an image.
*   **Preset and Custom Models**: Switch between built-in presets or use manual model selection for supported generation backends.
*   **Turbo and LoRA Controls**: Use Turbo LoRA cards and a general LoRA Stack directly from the generation panel.
*   **Canvas Editing Tools**: Move, transform, resize, snap, undo/redo, and manage generation results without leaving the node.
*   **Progress and Result Handling**: Track generation progress and apply results back into the canvas as editable layers.

## Jakkanna Pose Studio

<p align="center">
  <img src="images/pose-studio-logo.png" alt="Jakkanna Pose Studio" width="360">
</p>

**Example Workflows:** [QWEN](workflows/Jakkanna%20Pose%20Studio%20QWEN.json) · [Klein9b](workflows/Jakkanna%20Pose%20Studio%20Klein9b.json)

**Jakkanna Pose Studio** is a professional 3D posing, framing, lighting, and pose-library environment running directly inside a ComfyUI node. It is designed for building high-quality pose/control references without leaving the graph: adjust the character body, pose bones interactively, frame the camera, tune lights, manage saved poses, and output single images or pose batches.

### Key Features

*   **Interactive 3D Viewport**: Pose the mannequin directly in the node with selectable joints, bone manipulation, transform controls, and full **Undo/Redo** support.
*   **Dynamic Body Generator**: Fine-tune the character shape with sliders for Age, Gender blending, Weight, Muscle, and Height.
*   **Multi-Pose Tabs**: Create multiple independent pose states inside one node, making batch outputs and pose sequences easier to build.
*   **Pose Copy/Paste**: Transfer complex poses between tabs without rebuilding them from scratch.
*   **Modal Pose Gallery**: Save, browse, load, and delete poses in a focused full-screen gallery instead of cluttering the main workspace.
*   **Pose Import/Export**: Batch save and load pose data via JSON for reuse across workflows or projects.
*   **Tracing Support**: Load a background reference image and align the 3D character to it for accurate pose matching.
*   **Precision Camera Controls**: Set output dimensions, zoom, model rotation, and camera orbit with an integrated radar-style control.
*   **Viewport Frame Preview**: Preview the final render boundary directly in the viewport so composition matches the output.
*   **Advanced Environment Lighting**: Control Ambient, Directional, and Point Lights, including 2D radar controls for positioning point lights and radius controls for their influence.
*   **Keep Original Lighting Mode**: Bypass synthetic lighting for clean flat renders, useful for ControlNet-style pose/reference outputs.
*   **Prompt-Aware Lighting Output**: Generate descriptive lighting prompts that can be combined with your scene prompt.
*   **Custom Prompt Templates**: Use tag-based templates such as `<lighting>` and `<user_prompt>` to control how the final prompt is assembled.
*   **Direct Sidebar Prompting**: Add scene details in an auto-expanding prompt field directly inside the Pose Studio UI.
*   **Flexible Export Modes**: Output poses as a list or as a grid, with configurable background color.

👉 **[Detailed Usage Guide](docs/JAKKANNA_POSE_STUDIO_USAGE.md)**

## Additional Nodes

### Jakkanna Visual Camera Control
**[Example Workflow](workflows/Jakkanna%20Visual%20Camera%20Control.json)**

An interactive node with a visual widget for controlling camera position. It is designed for intuitive angle control and prompt generation, especially for multi-angle LoRAs like **Qwen-Image-Edit-2511-Multiple-Angles**.

*   **Visual Widget**: Select azimuth and distance with the mouse.
*   **Elevation Slider**: Pick elevation from -30° to 60°.
*   **Trigger Word Toggle**: Enable or disable the `<sks>` trigger from the widget.

### Jakkanna QWEN Detailer
**[Example Workflow](workflows/Jakkanna%20QWEN%20Detailer.json)**

A QWEN-Image-Edit2511 detailer for enhancing detected regions such as faces, hands, and objects with vision-guided instructions.

*   **Smart Cropping**: Automatically squares crops and handles padding.
*   **Vision-Guided Enhancement**: Uses QWEN-generated instructions or user prompts.
*   **Drift Fix**: Helps keep the enhanced area aligned with the original composition.
*   **Quality of Life Tools**: Includes color matching, Poisson blending, and upscaling options.
*   **Inpainting Mode**: Supports mask-based editing and filling black areas.
*   **QWEN Options**: Supports QWEN-Image-Edit2511-specific options such as `distortion_fix` and `qwen_2511` mode.

### Jakkanna Model Manager & Selector
**[Example Workflow](workflows/Jakkanna%20Model%20Loader%20Showcase.json)**

A system for managing and selecting LoRAs and checkpoints directly in ComfyUI, with support for Civitai and HuggingFace.

#### Jakkanna Model Manager
The backend node that reads a HuggingFace-hosted `model_updater.json` and manages model downloads.

*   **Repo ID**: Point the manager to your HuggingFace model repository.
*   **Downloads**: Queue and download models in the background.
*   **Civitai Support**: Use API key authentication for restricted Civitai models.

👉 **[Configuration Guide: How to create your own model repo](docs/MODEL_MANAGER_GUIDE.md)**

#### Jakkanna Model Selector
The companion UI node for choosing models from the configured repository.

*   **Visual Card UI**: Shows model name, version, status, and description.
*   **Smart Search**: Opens a searchable modal model list.
*   **Status Indicators**: Shows Installed, Update Available, Missing, and Downloading states.
*   **One-Click Install/Update**: Install or update models directly from the selector.
*   **Universal Connection**: Outputs a standard relative path string compatible with standard ComfyUI nodes.

👉 **[Usage Guide: How to use Selector with Standard Loaders](docs/MODEL_SELECTOR_USAGE.md)**

### Jakkanna BBox Extractor

A helper node for extracting and visualizing crops when you need detected bounding-box regions without running a full face/detailer workflow.

## Installation

### Recommended: ComfyUI Manager

1. Open **ComfyUI Manager**.
2. Choose **Custom Nodes Manager**.
3. Search for **Jakkanna**.
4. Click **Install**.
5. Restart ComfyUI.

Remove the upstream `vnccs-utils` package first if it is already installed.

### Manual Installation

Open a terminal in your ComfyUI directory and run:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/teenu/ComfyUI-Jakkanna.git
cd ComfyUI-Jakkanna
pip install -r requirements.txt
```

Restart ComfyUI after installation.

## Lineage, Licensing, and Maintenance

The original VNCCS implementation remains attributed to MiuProject under the
MIT license. MakeHuman-derived code and assets, SAM 3D Body, MHR, Three.js,
and other bundled components retain their own licenses and notices; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the package-level
inventory.

Jakkanna is maintained independently at
[teenu/ComfyUI-Jakkanna](https://github.com/teenu/ComfyUI-Jakkanna).
It is not affiliated with or endorsed by the upstream VNCCS project. Please
report Jakkanna defects through its
[issue tracker](https://github.com/teenu/ComfyUI-Jakkanna/issues).
