<p align="center">
  <img src="docs/assets/impossible-ocr-header.png" alt="Impossible G — OCR" width="860">
</p>

# Impossible OCR

Impossible OCR is a local-first OCR server. Version 0.1 is deliberately scoped to synchronous OCR
of static PNG or JPEG raster images with the curated PP-OCRv5 mobile detector and English mobile
recognizer bundle. HTTP, unary gRPC, and MCP all use the same bounded pipeline.

The server starts on loopback by default. Liveness is available immediately, while readiness stays
false and OCR requests return a stable `unavailable` error unless the optional ONNX backend is
enabled, its exact local bundle is installed, its integrity-bound native runtime is admitted, and
every session passes warm-up. This fail-closed behavior is intentional.

## Workspace

- `impossible-ocr-domain`: validated raster input, options, errors, geometry, and canonical results.
- `impossible-ocr-protocol`: transport-neutral HTTP, gRPC, and MCP boundary contracts.
- `impossible-ocr-pipeline`: backend lifecycle and synchronous pipeline traits.
- `impossible-ocr-onnx`: explicit model lifecycle, bounded graph admission, and optional CPU ONNX
  Runtime adapter.
- `impossible-ocr-server`: CLI plus bounded HTTP control plane and transport extension modules.
- `vendor/impossible-server`: provisional reviewed source snapshot of the neutral server foundation.

## Documentation

- [Product contract](docs/product-contract.md)
- [Model admission boundary](docs/model-admission.md)
- [Native ONNX Runtime packaging](docs/onnxruntime-packaging.md)
- [Deferred work and cross-project lessons](docs/backlog.md)

The backlog records possible follow-up work only. It does not block the current v0.1 scope and does
not commit the project to dates or features.

## Development

```text
cargo fmt --all -- --check
cargo check --locked --workspace --all-targets
cargo clippy --locked --workspace --all-targets -- -D warnings
cargo test --locked --workspace
cargo doc --locked --workspace --no-deps
pwsh ./scripts/verify-dependency-policy.ps1
pwsh ./scripts/verify-model-contracts.ps1
pwsh ./scripts/install-onnxruntime.tests.ps1
pwsh ./scripts/runtime-qualification-attestation.tests.ps1
pwsh ./scripts/verify-runtime-qualification-attestation.ps1
pwsh ./scripts/privacy-scan.tests.ps1
pwsh ./scripts/privacy-scan.ps1
pwsh ./scripts/verify-foundation-snapshot.ps1
cargo deny check advisories bans licenses sources
```

## Local ONNX backend

The default build contains no native ONNX Runtime dependency and remains an unavailable server
shell. Build with `--features onnx-runtime` to enable the CPU adapter. Startup remains strictly
offline: it never scans the machine, searches `PATH`, or downloads a model or runtime. The model
bundle must already exist in an explicit model store, and the native library must be supplied with
its exact byte length and lowercase SHA-256.

An operator can populate the store from the four already-downloaded, pinned PaddlePaddle files. The
command performs no network access and emits only a fixed JSON status object:

```text
cargo run --locked -p impossible-ocr-server --bin import-model-bundle -- \
  --model-store <absolute-model-store> \
  --detector-graph <detector-inference.onnx> \
  --detector-config <detector-inference.yml> \
  --recognizer-graph <recognizer-inference.onnx> \
  --recognizer-config <recognizer-inference.yml>
```

```text
cargo run --locked -p impossible-ocr-server --features onnx-runtime \
  --bin impossible-ocr-server -- \
  --model-store /opt/impossible-ocr/models \
  --runtime-directory /opt/impossible-ocr/runtime \
  --runtime-library libonnxruntime.so.1.28.0 \
  --runtime-library-bytes <exact-native-library-size> \
  --runtime-library-sha256 <exact-native-library-sha256> \
  --runtime-lanes 1 --runtime-intra-threads 1 --runtime-inter-threads 1
```

Only the curated `paddlex-ocr-3.7-max960` profile is accepted. Runtime API 28 is compiled in. The
official Windows x64 and Linux x64 ONNX Runtime 1.28.0 archives and their extracted redistribution
files are packaging-qualified and allowlisted. Windows inference qualification is bound to the
checked-in privacy-safe attestation produced by a fresh real run. Linux archive inspection does not
execute Linux binaries, so Linux is packaging-qualified but inference-unqualified until the same
harness passes natively on Linux. No weights or native runtime binaries are committed here.

The Windows attestation contains only pinned public identities and fixed boolean probe outcomes. It
contains no paths, machine specifications, timings, tensors, or recognized text. Any relevant code,
model contract, or Windows runtime-record change invalidates it in CI.

Licensed under either Apache-2.0 or MIT, at your option.
