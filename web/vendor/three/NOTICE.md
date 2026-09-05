# three.js, vendored

WebGL renderer, scene graph, loaders and maths used by the Three.js nodes.

- Upstream: <https://github.com/mrdoob/three.js>
- Version r185, npm `three@0.185.0`
- Licence: MIT, retained verbatim in `LICENSE` beside this file

## Files taken from upstream

| File | Upstream path | Changed |
|---|---|---|
| `curves/NURBSCurve.js` | `examples/jsm/curves/NURBSCurve.js` | import specifiers |
| `curves/NURBSUtils.js` | `examples/jsm/curves/NURBSUtils.js` | import specifiers |
| `environments/RoomEnvironment.js` | `examples/jsm/environments/RoomEnvironment.js` | import specifiers |
| `libs/fflate.module.js` | `examples/jsm/libs/fflate.module.js` | no |
| `lights/RectAreaLightTexturesLib.js` | `examples/jsm/lights/RectAreaLightTexturesLib.js` | import specifiers |
| `lights/RectAreaLightUniformsLib.js` | `examples/jsm/lights/RectAreaLightUniformsLib.js` | import specifiers |
| `loaders/3MFLoader.js` | `examples/jsm/loaders/3MFLoader.js` | import specifiers |
| `loaders/ColladaLoader.js` | `examples/jsm/loaders/ColladaLoader.js` | import specifiers |
| `loaders/EXRLoader.js` | `examples/jsm/loaders/EXRLoader.js` | import specifiers |
| `loaders/FBXLoader.js` | `examples/jsm/loaders/FBXLoader.js` | import specifiers |
| `loaders/GLTFLoader.js` | `examples/jsm/loaders/GLTFLoader.js` | import specifiers |
| `loaders/HDRLoader.js` | `examples/jsm/loaders/HDRLoader.js` | import specifiers |
| `loaders/MTLLoader.js` | `examples/jsm/loaders/MTLLoader.js` | import specifiers |
| `loaders/OBJLoader.js` | `examples/jsm/loaders/OBJLoader.js` | import specifiers |
| `loaders/PLYLoader.js` | `examples/jsm/loaders/PLYLoader.js` | import specifiers |
| `loaders/STLLoader.js` | `examples/jsm/loaders/STLLoader.js` | import specifiers |
| `loaders/TGALoader.js` | `examples/jsm/loaders/TGALoader.js` | import specifiers |
| `loaders/collada/ColladaComposer.js` | `examples/jsm/loaders/collada/ColladaComposer.js` | import specifiers |
| `loaders/collada/ColladaParser.js` | `examples/jsm/loaders/collada/ColladaParser.js` | import specifiers |
| `postprocessing/BokehPass.js` | `examples/jsm/postprocessing/BokehPass.js` | import specifiers |
| `postprocessing/EffectComposer.js` | `examples/jsm/postprocessing/EffectComposer.js` | import specifiers |
| `postprocessing/MaskPass.js` | `examples/jsm/postprocessing/MaskPass.js` | import specifiers |
| `postprocessing/OutputPass.js` | `examples/jsm/postprocessing/OutputPass.js` | import specifiers |
| `postprocessing/Pass.js` | `examples/jsm/postprocessing/Pass.js` | import specifiers |
| `postprocessing/RenderPass.js` | `examples/jsm/postprocessing/RenderPass.js` | import specifiers |
| `postprocessing/SMAAPass.js` | `examples/jsm/postprocessing/SMAAPass.js` | import specifiers |
| `postprocessing/ShaderPass.js` | `examples/jsm/postprocessing/ShaderPass.js` | import specifiers |
| `postprocessing/UnrealBloomPass.js` | `examples/jsm/postprocessing/UnrealBloomPass.js` | import specifiers |
| `shaders/BokehShader.js` | `examples/jsm/shaders/BokehShader.js` | import specifiers |
| `shaders/CopyShader.js` | `examples/jsm/shaders/CopyShader.js` | import specifiers |
| `shaders/LuminosityHighPassShader.js` | `examples/jsm/shaders/LuminosityHighPassShader.js` | import specifiers |
| `shaders/OutputShader.js` | `examples/jsm/shaders/OutputShader.js` | import specifiers |
| `shaders/SMAAShader.js` | `examples/jsm/shaders/SMAAShader.js` | import specifiers |
| `three.core.js` | `build/three.core.js` | no |
| `three.module.js` | `build/three.module.js` | no |
| `utils/BufferGeometryUtils.js` | `examples/jsm/utils/BufferGeometryUtils.js` | import specifiers |
| `utils/SkeletonUtils.js` | `examples/jsm/utils/SkeletonUtils.js` | import specifiers |

Every file marked "import specifiers" imports `three` and `three/addons/...` upstream. Each of
those specifiers is rewritten to the relative path of the copy in this directory, which is why
the upstream `examples/jsm/` layout is mirrored here rather than flattened. Nothing else in any
file is altered.

`three.module.js` re-exports `three.core.js`, so both files are required and both are loaded
from this directory. Nothing is fetched over the network at runtime.

## fflate, bundled inside three.js

`libs/fflate.module.js` is fflate 0.8.2 by Arjun Barrett, MIT, carried inside the three.js
distribution and taken with it. Its licence and copyright are in the banner at the top of the
file. `FBXLoader` and `3MFLoader` read compressed data through it.

- Upstream: <https://github.com/101arrowz/fflate>

## Not taken

`examples/jsm/controls/OrbitControls.js` is not taken. Camera control is `createOrbitControls`
in `web/threejs/runtime.js`.
