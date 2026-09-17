# Model admission boundary

The v0.1 backend is the official PP-OCRv5 mobile detector plus English mobile recognizer executed on
the ONNX Runtime CPU provider when the optional runtime feature is enabled. This repository does
not vendor model weights or a native runtime. Model installation is an explicit, integrity-checked
operation; normal startup remains offline. The `import-model-bundle` operator command accepts only
four explicit local paths for the curated artifacts, verifies the pinned sizes and SHA-256 values,
validates the bounded Paddle configuration and ordered dictionary, and atomically promotes the
complete bundle. It has no download or host-scanning mode.

The lifecycle manifest pins the full immutable upstream revision, filename, exact byte length,
SHA-256, license evidence, preprocessing constants, postprocessing profile, and recognition
dictionary material for both artifacts. The two artifacts form one atomic bundle; a partial or
mixed-version bundle is invalid.

The checked-in graph contracts are qualified from the two exact, integrity-bound upstream ONNX
artifacts. They record the exact artifact byte length and SHA-256 together with the measured ONNX IR
version, opsets, tensor signatures, and recursive operator multiset. The runtime-independent
`qualify-onnx` tool takes only an explicit artifact path and curated role, verifies the pinned byte
length and SHA-256 before invoking the bounded parser, and emits deterministic JSON without local
paths or host metadata. It never searches for or downloads model weights.

The bounded protobuf admission layer rejects non-standard operator domains, external-data
references (including nested attribute graphs and tensors), model-local functions, training graphs,
malformed or duplicate graph I/O, unknown protobuf fields, structural-budget overruns, digest
mismatches, and any difference from a qualified contract. The current root workspace targets Rust
1.88 and pins optional `ort 2.0.0-rc.13`/`ort-sys 2.0.0-rc.13`. The dependency enables only `std`,
`load-dynamic`, and the reviewed `api-28` ABI selector; default builds exclude it. Feature-enabled
session construction accepts only an explicit absolute native-library path with pinned byte length
and SHA-256. The library is streamed into owned content-addressed storage, reverified, and held
against replacement while loaded. No configuration searches for or downloads a runtime. The
official Windows x64 and Linux x64 ONNX Runtime 1.28.0 archives, extracted runtime, provider
sidecar, license, notices, and privacy files are integrity-qualified and allowlisted. Linux archive
inspection never executes Linux binaries; platform-native runtime qualification remains a release
gate.

The `vendor/impossible-server` directory is a frozen foundation snapshot excluded from the root
workspace. Its historical toolchain and dependency metadata document that vendored foundation, not
the current OCR workspace's Rust 1.88 and ORT rc.13 policy, and must not be rewritten during this
migration.
