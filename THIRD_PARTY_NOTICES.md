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

The `vnccs_sam3d/` directory vendors the SAM 3D Body bridge and related MHR
integration under multiple licenses:

* wrapper and integration code under MIT;
* SAM 3D Body under Meta's SAM License; and
* the vendored DINOv3 subtree under Meta's DINOv3 License; and
* Momentum Human Rig components under Apache License 2.0.

Redistributions must retain the applicable license and attribution files in
`vnccs_sam3d/licenses/`, including `LICENSE-SAM`, `LICENSE-MHR`,
`NOTICE-MHR`, and `THIRD_PARTY_NOTICES`. The DINOv3 license is retained at
`vnccs_sam3d/sam_3d_body/models/backbones/dinov3_repo/LICENSE.md`.
The vendored notice also identifies MIT-licensed UniRig-derived integration
code. Blender is downloaded separately when explicitly requested and remains
under the GNU GPL version 3.

## Three.js and Browser-Side Dependencies

`web/three.module.js`, `web/OrbitControls.js`, and
`web/TransformControls.js` are from Three.js revision 160, Copyright
2010-2023 Three.js Authors, licensed under MIT. The bundled module retains
the copyright and SPDX notice.

## Python Dependencies

Dependencies listed in `pyproject.toml` and `requirements.txt` are distributed
by their respective authors under their own licenses. They are installed
separately and are not relicensed by this repository.
