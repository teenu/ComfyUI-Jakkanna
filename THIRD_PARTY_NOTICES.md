# Third-Party Notices

This repository contains components under multiple licenses. The root
[`LICENSE`](LICENSE) applies to the original VNCCS code and MIT-licensed
Jakkanna contributions where no component-specific notice states otherwise.
It does not replace the licenses of bundled or derived third-party components.
The repository as distributed is therefore not MIT-only.

## MakeHuman

Source: [makehumancommunity/makehuman](https://github.com/makehumancommunity/makehuman)

* `CharacterData/matrix.py`, `CharacterData/transformations.py`, and portions
  of `CharacterData/mh_skeleton.py` are derived from MakeHuman source code and
  carry the GNU AGPL version 3 or any later version terms stated in those
  sources and the bundled MakeHuman license.
* The complete AGPLv3 text is provided in
  `CharacterData/makehuman/LICENSE.CODE.md`.
* The bundled MakeHuman meshes, targets, skeleton data, weights, and related
  graphical assets are released under CC0 1.0 Universal. See
  `CharacterData/makehuman/LICENSE.ASSETS.md`.
* `CharacterData/transformations.py` also retains the copyright,
  redistribution conditions, and warranty disclaimer of the BSD-licensed
  code from which it was adapted.

## SAM 3D Body, DINOv3, and Momentum Human Rig

The `jakkanna_sam3d/` directory vendors the SAM 3D Body bridge and related MHR
integration under multiple licenses:

* wrapper and integration code under MIT;
* SAM 3D Body under Meta's SAM License; and
* the vendored DINOv3 subtree under Meta's DINOv3 License; and
* Momentum Human Rig components under Apache License 2.0.

Redistributions must retain the applicable license and attribution files in
`jakkanna_sam3d/licenses/`, including `LICENSE-SAM`, `LICENSE-MHR`,
`NOTICE-MHR`, and `THIRD_PARTY_NOTICES`. The DINOv3 license is retained at
`jakkanna_sam3d/sam_3d_body/models/backbones/dinov3_repo/LICENSE.md`.
The vendored notice also identifies MIT-licensed UniRig-derived integration
code. Blender is downloaded separately when explicitly requested and remains
under the GNU GPL version 3.

## Three.js and Browser-Side Dependencies

`web/three.module.js`, `web/OrbitControls.js`, and
`web/TransformControls.js`, together with the FBX and NURBS modules under
`web/vendor/three-r160/`, are from Three.js revision 160, Copyright 2010-2023
Three.js Authors, licensed under MIT. The bundled modules retain copyright
and SPDX notices.

`web/vendor/three-r160/libs/fflate.module.js` is fflate 0.6.9, Copyright
Arjun Barrett, licensed under MIT. Its source header retains the upstream
license notice.

`web/vendor/ag-psd.bundle.mjs` and `web/vendor/ag-psd.real.mjs` bundle
ag-psd 28.2.2, Copyright Agamnentzar, licensed under MIT. The bundled browser
Buffer implementation under `web/vendor/node/` retains its MIT and BSD
license notices in the source.

## SCAIL-2 Flow-UniPC

`nodes/scail2_flow_unipc.py` is copied from the Flow-UniPC implementation in
`zai-org/SCAIL-2` commit `5cfe1b8daac8bcb22ee19794e6c04f1bf5de6ac5`, which
is released under Apache License 2.0. That implementation is derived from the
Diffusers 0.31.0 UniPC scheduler and the Alibaba Wan Team flow-matching
conversion identified in its source header. Diffusers is Copyright 2024 The
Hugging Face Team and licensed under Apache License 2.0. The complete Apache
License 2.0 text is retained at `jakkanna_sam3d/licenses/LICENSE-MHR`.

## Python Dependencies

Dependencies listed in `pyproject.toml` and `requirements.txt` are distributed
by their respective authors under their own licenses. They are installed
separately and are not relicensed by this repository.
