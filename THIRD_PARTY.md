# Third-party inventory

Inspected 2026-10-27.

This is an inventory of what the repository files declare, **not a legal
determination** or a statement that every asset may be redistributed. Code,
assets and notices are inherited unchanged from upstream PufferLib 5
(`ea7f989371270d2948f96728bda8aec5e7634a4c`) unless noted below. The Metal
adaptations and robot-arm asset conversion are fork additions. Dependency pins
were recorded during cleanup leg 2. This inventory does not change any license.
The project-level MIT notice is in [LICENSE](LICENSE); it must not be read as
replacing more specific notices or filling missing third-party provenance.

“Not stated in repository” means no specific statement was found in the tracked
files (including embedded font metadata). Downloaded dependency notices below
are identified separately: those files exist after the corresponding build and
are not part of the tracked source distribution. URLs identify sources where
known; they do not assert licensing terms absent from the files.

## Vendored code and derived implementations

| Files/component | Source/version stated or identifiable | License declared; notice location |
|---|---|---|
| `vendor/cJSON.{c,h}` | [cJSON](https://github.com/DaveGamble/cJSON), 1.7.19, Dave Gamble and contributors | MIT text in both file headers |
| `vendor/collections-c/include/{cc_array.h,cc_common.h}` | Collections-C, Srđan Panić; source URL/version not stated in repository | LGPL version 3 or later; both header notices (no full LGPL text bundled) |
| `vendor/collections-c/include/dlmalloc.h` | Doug Lea, malloc 2.7.2, last update 2001-02-25; header links to [Doug Lea's malloc description](http://gee.cs.oswego.edu/dl/html/malloc.html) | Public domain; header notice |
| `vendor/KHR/khrplatform.h` | [Khronos EGL Registry](https://github.com/KhronosGroup/EGL-Registry), last semantic change `67a3e0864c2d75ea5287b9f3d2eb74a745936692` | Khronos permissive permission/warranty text; header |
| `vendor/glad.h` | [glad generator](https://glad.dav1d.de/), 0.1.36, GL 3.3 core, generated 2025-10-14 | License not stated in repository for the generated header; generator details in header; Khronos header has its separate notice above |
| `vendor/raygui.h` (including embedded icons) | [raygui](https://github.com/raysan5/raygui), 4.5-dev | zlib/libpng; header notice, Ramon Santamaria |
| `vendor/rcamera.h` | [raylib](https://github.com/raysan5/raylib), camera helper | zlib/libpng; header notice, Christoph Wagner and Ramon Santamaria |
| `vendor/rlights.h`, `ocean/battle/rlights.h` | raylib lighting helper, Victor Fisac and Ramon Santamaria | zlib/libpng; both header notices |
| `vendor/minshell.html` | raylib web shell (raylib links in file); also loads [FileSaver.js](https://github.com/eligrey/FileSaver.js) from jsDelivr without a version pin | Specific shell and FileSaver license not stated in repository; no separate notice |
| `vendor/puf_web_vsync.js` | PufferLib Emscripten/Asyncify glue | No separate notice; project [LICENSE](LICENSE); separate third-party provenance not stated in repository |
| `vendor/fast-nle/` | Downloaded, not tracked; see dependency table below | Downloaded notices, not a vendored license grant from this repository |
| `ocean/{battle,nmmo3,terraform}/simplex.h` | Casey Duncan noise library (2008), Zev Benjamin vec_noise (2017), Joseph Suarez C port (2024); noise-library source URL not stated; sine/cosine approximation links to a devmaster.net discussion | MIT as stated in each header |
| `ocean/drone/{drone.h,dronelib.h,render.h}` | [tensaur/drone](https://github.com/tensaur/drone), Sam Turner and Finlay Sanders (2025); `dronelib.h` also cites [learning-to-fly](https://github.com/arplaboratory/learning-to-fly) | Original project's MIT license stated in all three headers; no separate license stated for cited learning-to-fly material |
| `ocean/craftax/{craftax.h,craftax_parity.h,constants.h}` | Craftax/JAX port; `craftax.h` describes relationship to original mechanics | Separate upstream license and source URL not stated in repository; provenance comment in `craftax.h` |
| `ocean/craftax_classic/craftax_classic.h` | [Infatoshi/craftax.c](https://github.com/Infatoshi/craftax.c), Matthews et al. 2024 Craftax-Classic | Separate derived-source license not stated in repository; provenance in header |
| `ocean/guerrillacheckers/mcts.h` | `nico/guerrillacheckers/src/mcts.nim` named in header; source URL not stated in repository | Separate derived-source license not stated in repository |
| `ocean/wef/wef.h` | [KempnerInstitute/wef](https://github.com/KempnerInstitute/wef) biophysics port named in header | Separate derived-source license not stated in repository |
| `ocean/robocode/agent_drussgt.h` | DrussGT 3.1.4159, Julian Kent / Skilgannon | Header says “keep open-source; credit Skilgannon / DrussGT”; no standard license text or source URL stated. Referenced `third_party/` jar is not tracked here |
| `ocean/robocode/bots.h`, `agent_hawk_on_fire.h`, `agent_raiko.h` | BeepBoop adaptation cites [Understanding BeepBoop](https://robowiki.net/wiki/BeepBoop/Understanding_BeepBoop); HawkOnFire and Raiko bot names | Separate derived-source licenses and precise revisions not stated in repository; BeepBoop provenance in `bots.h` |
| `ocean/impulse_wars/helpers.h` (`wyhash64`) | [Daniel Lemire RNG article](https://lemire.me/blog/2019/03/19/the-fastest-conventional-random-number-generator-that-can-pass-big-crush/) and [lemire/testingRNG](https://github.com/lemire/testingRNG) cited above function | Separate derived-source license not stated in repository |
| `ocean/onestateworld/onestateworld.h` (Marsaglia polar method) | [Wikipedia algorithm reference](https://en.wikipedia.org/wiki/Marsaglia_polar_method) cited above implementation | Separate snippet license not stated in repository |
| `ocean/nethack/` | fast-nle adapter; [README](ocean/nethack/README.md) | No separate adapter notice; project LICENSE; downloaded NetHack notice listed below |
| `ocean/overcooked/` | [README](ocean/overcooked/README.md) describes game inspiration | Separate original-code/art license and source URL not stated in repository |
| `ocean/freeway/freeway.h` | Cites [ALE Freeway](https://ale.farama.org/environments/freeway/) | Separate source license not stated in repository; reference in header |
| `ocean/osrs/`, `ocean/osrs_{colosseum,inferno,pvp,zulrah}/` | OSRS environments; asset download described below | Separate game/asset rights not stated in repository; project-level code LICENSE does not resolve them |
| `src/metal/shaders/rollout.metal`, `src/metal/compat/curand_kernel.h`, `src/metal/protein_cpu.h` (**fork**) | cuRAND-compatible Philox4x32-10 state/generator and CPU compatibility implementations; described in file comments | No separate third-party source license or source URL stated in repository; project LICENSE; API compatibility alone is not a claim of copied NVIDIA source |
| `tools/metal/joe-kuo/new-joe-kuo-6.20000`, `src/metal/protein_sobol_public.bin` (**fork data**) | Exact publisher-text excerpt and compact first 20,000 dimensions of Joe--Kuo `new-joe-kuo-6.21201`; generator `tools/metal/gen_protein_sobol.py`; fixed SplitMix64 digital shifts | Frances Y. Kuo and Stephen Joe, BSD-style license in `tools/metal/joe-kuo/LICENSE`, reproduced below and in [NOTICE](NOTICE) |
| `src/metal/runtime.mm`, `src/metal/shaders/gemm.metal` (**fork adaptations**) | [Valtteri Valo/PufferLib](https://github.com/valtterivalo/PufferLib), `metal-local`, upstream PR #532; GEMM comment references `1e7e9dd1` | MIT; [NOTICE](NOTICE) describes adaptation scope |
| `src/metal/shaders/{muon.metal,wide_muon.h}` (**fork adaptations**) | [MLX](https://github.com/ml-explore/mlx), `56c2bf9236d929097f61e2c6e55be4e4bb6da249`, shuffle reduction and fragment coordinates | MIT; Apple copyright in shader comments and full notice/adaptation scope in [NOTICE](NOTICE) |

### Protein Sobol data provenance

Metal's default Protein source is the public Joe--Kuo D(6) data, attributed to
Frances Y. Kuo and Stephen Joe, *Constructing Sobol sequences with better
two-dimensional projections*, SIAM J. Sci. Comput. 30 (2008), 2635--2654.
Source: https://web.maths.unsw.edu.au/~fkuo/sobol/ and the
[Joe--Kuo repository](https://github.com/joe-kuo/sobol_data).
The [license](https://github.com/joe-kuo/sobol_data/blob/master/LICENSE) applies
to `sobol.cc` and the accompanying direction-number sets.

`tools/metal/gen_protein_sobol.py` verifies the complete 1,887,612-byte source
text SHA256
`68eedd2a4e3b659b9695e7aff0f8ac68718bcf620730fc3d3a8c65df2a067441`, derives
its header and rows through dimension 20,000, and checks that result byte for
byte against the committed 1,771,202-byte excerpt. The excerpt SHA256 is
`af7492ff6eb0c4e1d968a15710d6496e61709d151821859841b0c16893d7b905`.
Offline generation uses that excerpt directly and packs exactly Protein's
existing supported limit, including its implicit dimension 1. The 394,264-byte
committed artifact has SHA256
`adca16e3a020a0ee099fd130b6cb5f46628c14fe629909b7fa850eb585d07210`.
At first use, the standard 32-bit recurrence expands it into 32 direction words
per dimension. A digital XOR shift uses the low 32 bits of consecutive
SplitMix64 outputs from fixed seed `0x50554646534f424c` (`PUFFSOBL`). Neither the
direction data nor shifts derive from cuRAND. Builds and tests work offline;
the public gate regenerates the artifact from the excerpt and includes a
mutated-direction negative control.

The former NVIDIA-exported table is removed from the tracked tree. Optional
exact CUDA mode reads a user-supplied external fixture, with strict size and
SHA256 checks; see [fixture instructions](tests/metal_ref/README-protein.md).
Default sweeps retain the scrambled Sobol statistical construction but their
suggestions are no longer bitwise-identical to CUDA. This is a digital shift,
not a reproduction of cuRAND's scrambled-direction transform.

```text
-----------------------------------------------------------------------------
Licence pertaining to sobol.cc and the accompanying sets of direction numbers

-----------------------------------------------------------------------------
Copyright (c) 2008, Frances Y. Kuo and Stephen Joe
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright
      notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above copyright
      notice, this list of conditions and the following disclaimer in the
      documentation and/or other materials provided with the distribution.

    * Neither the names of the copyright holders nor the names of the
      University of New South Wales and the University of Waikato
      and its contributors may be used to endorse or promote products derived
      from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```

Other `ocean/*` environments are supplied as upstream PufferLib code under the
project-level notice. No additional third-party code origin or license is
stated in their files; game names or similarity alone are not evidence of a
particular source-code license. The asset inventory below applies separately.

## Resources and fonts

Every tracked `resources/*` top-level set is listed here. An absent separate
notice is recorded as such, rather than assigning the project's MIT license to
unknown artwork. Paths include nested shaders, textures, models, data, trained
weights and media, except for the explicitly separated fonts.

| Set | Source/provenance and license declared | Notice file |
|---|---|---|
| `resources/battle/` | Model and shader origin/license not stated in repository | None |
| `resources/blastar/` | Sprite origin/license not stated in repository | None |
| `resources/boxoban/` | Texture origin/license not stated in repository | None |
| `resources/chess/pieces.png` | [“Chess Pieces Sprite”](https://commons.wikimedia.org/wiki/File:Chess_Pieces_Sprite.svg), Jurgenwesterhof, adapted from Cburnett; unmodified PNG reproduction; CC BY-SA 3.0 | `ocean/chess/chess.h` near texture loading; links to source and https://creativecommons.org/licenses/by-sa/3.0/ |
| `resources/constellation/` | Shader/config origin and separate license not stated in repository | None |
| `resources/cpr/` | Image origin/license not stated in repository | None |
| `resources/craftax/` | Texture source/license not stated in repository | None |
| `resources/dino/` | Sprite origin/license not stated in repository | None |
| `resources/drive/` | `map_{010,942}.bin` are produced from the [preprocessed Waymo Open Motion Dataset](https://huggingface.co/datasets/daphne-cornelisse/pufferdrive_womd_train_1000) by `ocean/drive/dataset.py`; car-model source/license and separate map-data license are not stated in repository | Dataset source and conversion instructions in `ocean/drive/dataset.py`; no asset notice |
| `resources/drone/` | Crazyflie model/Blender source license not stated in repository; drone code attribution does not specifically license these assets | None |
| `resources/enduro/` | Sprite source/license not stated in repository | None |
| `resources/freeway/` | Texture source/license not stated in repository | None |
| `resources/impulse_wars/` | `wall_texture_map.png` is generated by the included `create_texture.py`; bloom/blur shaders declare Le Juez Victor (2025) and a zlib-style permission notice, and blur coefficients cite a [generator](https://lisyarus.github.io/blog/posts/blur-coefficients-generator.html); separate origin/license for the remaining shaders is not stated | Notices in both GLSL 100/330 bloom and blur files; no directory-wide notice |
| `resources/laser_puzzle/` | Level data origin/license not stated in repository | None |
| `resources/moba/` | Map/art/shader source/license not stated in repository | None |
| `resources/nethack/msgcode128.bin` | Embedding/pretrained data source/license not stated in repository | None |
| `resources/nmmo3/` | Mana Seed by Seliel the Shaper, purchased from itch.io; artist terms apply; notice prohibits repurposing for other projects without an independent license | [ASSETS_LICENSE.md](resources/nmmo3/ASSETS_LICENSE.md), also `ocean/nmmo3/nmmo3.h` |
| `resources/onlyfish/` | Star image source/license not stated in repository | None |
| `resources/overcooked/` | Sprite/texture source/license not stated in repository | None |
| `resources/pacman/` | Sprite source/license not stated in repository | None |
| `resources/robocode/` | Image source/license not stated in repository | None |
| `resources/robot_arm/franka_panda.glb` (**fork asset conversion**) | [MuJoCo Menagerie Franka Emika Panda](https://github.com/google-deepmind/mujoco_menagerie/tree/main/franka_emika_panda); Apache 2.0; conversion documented as `tools/pack_panda_glb.py` (that helper is not tracked here) | [ATTRIBUTION.md](resources/robot_arm/ATTRIBUTION.md), [PANDA_LICENSE](resources/robot_arm/PANDA_LICENSE) |
| `resources/robot_arm/panda_lighting.{fs,vs}` (**fork**) | Renderer shaders; no separate third-party notice | Project LICENSE |
| `resources/shared/JetBrainsMono-{Bold,Medium,Regular,SemiBold}.ttf` | [JetBrains Mono](https://github.com/JetBrains/JetBrainsMono); copyright 2020 project authors; SIL Open Font License 1.1 | Each TTF's embedded `name` table, IDs 0, 13 and 14; no standalone OFL file |
| `resources/shared/Montserrat-Regular.ttf` | [Montserrat](https://github.com/JulietaUla/Montserrat); copyright 2011 project authors; SIL Open Font License 1.1 | TTF embedded `name` table, IDs 0, 13 and 14; no standalone OFL file |
| `resources/shared/Roboto-{Italic,Regular}.ttf` | [Roboto Classic](https://github.com/googlefonts/roboto-classic); copyright 2011 project authors; SIL Open Font License 1.1 (as embedded in these particular files) | Each TTF's embedded `name` table, IDs 0, 13 and 14; no standalone OFL file |
| Remaining `resources/shared/` | Puffer model/images source/license not stated in repository | None |
| `resources/terraform/` | Model/texture/shader source/license not stated in repository | None |
| `resources/tower_climb/` | GLSL 100 vertex shader identifies code from [glsl-inverse](https://github.com/glslify/glsl-inverse) and [glsl-transpose](https://github.com/glslify/glsl-transpose); maps, models, texture, remaining shader origin, and separate licenses are not stated in repository | Source links in `shaders/gls100/lighting.vs`; no directory-wide notice |
| `resources/tower_defence/` | Kbediako-created artwork, permission under repository MIT license; learned policy provenance and file manifest recorded | [ASSETS.md](resources/tower_defence/ASSETS.md) |
| `resources/trailer/` | Video, images, sounds, model/data and trailer sources; separate origins/licenses not stated in repository | None |

## Downloaded build dependencies

These are generated/downloaded caches, not tracked vendor contents. The main
archive URLs, versions and six SHA-256 values are pinned in
[tools/build_dependencies.sh](tools/build_dependencies.sh). Existing downloaded
notices are reported here as observed, without importing new license claims
into the tracked tree. System tools (Xcode/Metal, libomp, CMake, flex, bison,
Python/NumPy, CUDA/NCCL and Emscripten where used) are externally supplied
prerequisites, not vendored libraries or pinned downloads of this build.

| Dependency | Source and pin | Declared license and notice location after download |
|---|---|---|
| raylib | [raylib 5.5](https://github.com/raysan5/raylib/releases/tag/5.5), macOS/Linux/web archives and hashes in dependency helper | zlib/libpng in `raylib-5.5_*/include/raylib.h` |
| Box2D | [capnspacehook/box2d df25d747](https://github.com/capnspacehook/box2d/releases/tag/df25d747), macOS ARM64/Linux AMD64/web archives and hashes in helper | MIT SPDX identifier in downloaded `box2d-*/include/box2d/base.h`; no full Box2D notice tracked in repository |
| fast-nle / NetHack | [FinlaySanders/fast-nle](https://github.com/FinlaySanders/fast-nle), `e7f6568568242745a7f7ebd8f730367123522cac` | NetHack General Public License in downloaded `vendor/fast-nle/LICENSE`; individual source copyrights remain in downloaded files |
| pybind11 (fast-nle transitive CMake dependency) | [pybind/pybind11](https://github.com/pybind/pybind11), `v3.0.1` in downloaded `vendor/fast-nle/CMakeLists.txt` | BSD-style permission notice in `vendor/fast-nle/build/_deps/pybind11-src/LICENSE` |
| deboost.context / Boost.Context assembly (fast-nle transitive CMake dependency) | [septag/deboost.context](https://github.com/septag/deboost.context); CMake declares `GIT_HASH 259fc4103bad6bb484d5ff426ace56ac557107a4`, **not** `GIT_TAG`; observed downloaded HEAD `9010551e008e8cbaf23925d05aa50a3efb7d2348` differs, so this is not recorded as a verified effective pin | MIT in downloaded `vendor/fast-nle/build/_deps/deboost_context-src/LICENSE`; assembly headers separately declare Boost Software License 1.0 |
| bzip2/libbzip2 (fast-nle transitive CMake dependency) | `git://sourceware.org/git/bzip2.git`; CMake declares `GIT_HASH fbc4b11da543753b3b803e5546f56e26ec90c2a7`, not `GIT_TAG`; downloaded header identifies 1.0.8 (2019-07-13); observed downloaded HEAD `f5dfc96063fef2d9c5446d0733867d899ef9b942` differs from the declaration; not recorded as a verified effective pin | Julian Seward's four-condition permission/warranty text in downloaded `vendor/fast-nle/build/_deps/bzip2-src/LICENSE` |
| libtmt (inside fast-nle) | Rob King, 2017, Tiny Mock Terminal; downloaded README notes hackrl modifications; pinned with fast-nle; separate source URL not stated | BSD-style three-condition notice in downloaded `vendor/fast-nle/third_party/libtmt/{tmt.c,tmt.h,README.rst}` |
| ttyrec converter (inside fast-nle) | Pinned with fast-nle; `pyconverter.cc` credits Facebook and affiliates; `stripgfx.c` credits NetHack Development Team 1992 | Downloaded `vendor/fast-nle/third_party/converter/stripgfx.c` refers to NetHack license; no separate converter-wide license/provenance statement in the tracked repository |
| OSRS assets | [osrs-assets-v22](https://github.com/valtterivalo/PufferLib/releases/tag/osrs-assets-v22), SHA-256 `2f908da5b5ddf148c0cbef48ad6334c5253c04c6cd83782340bc2f1c5dffc7ad` in `ocean/osrs/asset_manifest.json`; fetched by `ocean/osrs/scripts/setup-data.sh` to ignored `ocean/osrs/data/` (including RuneScape fonts, models, animations, atlases and data) | License/provenance not stated in repository; see [OSRS README](ocean/osrs/README.md) |
| FileSaver.js (web runtime fetch) | Unversioned jsDelivr GitHub URL in `vendor/minshell.html` | License not stated in repository; script is fetched at runtime, not bundled |

fast-nle also carries its own Python packaging/lock metadata in the downloaded
checkout. Those optional Python packages are not installed by the native
trainer's CMake build and are not vendored into this repository. This inventory
does not silently upgrade transitive pins or add missing license texts. Missing
provenance, complete license-text packaging and redistribution decisions remain
owner review items.
