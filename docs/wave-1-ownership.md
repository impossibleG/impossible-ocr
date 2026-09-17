# Wave 1 ownership and shared dependency freeze

Wave 1 uses one shared lockfile and deliberately separates feature-owned source paths. Developers
must not edit another feature's files or root dependency metadata during the wave. Dependency
changes are integration work and must be reviewed centrally before regenerating `Cargo.lock`.

## Shared integration files

The integration owner exclusively owns `Cargo.toml`, `Cargo.lock`, `deny.toml`, all crate
`Cargo.toml` files, and this document. The current root workspace pins Rust 1.88 and keeps image,
transport, and optional inference dependencies exact. It does not add OpenCV, FFmpeg, native
geometry libraries, or model weights. The frozen `vendor/impossible-server` foundation is excluded
from the root workspace; its historical toolchain and dependency metadata are not the current OCR
dependency policy and remain untouched.

## Feature A: raster and pure algorithms

Owned paths:

- `crates/impossible-ocr-domain/src/**`
- `crates/impossible-ocr-pipeline/src/**`
- tests and fixtures beneath those two crates

The prepared dependency is exactly `image 0.25.6` with default features disabled and only `png` and `jpeg`
enabled. Geometry, DB postprocessing, deterministic preprocessing, reading order, cropping, and CTC
decoding remain pure Rust product code.

## Feature B: verified bundle lifecycle

Owned paths:

- `crates/impossible-ocr-onnx/src/**`
- tests and fixtures beneath that crate
- model-lifecycle documentation that does not define transport behavior

The prepared graph provides bounded streaming HTTPS through rustls, SHA-256, URL validation,
filesystem locking, asynchronous filesystem I/O, Unix durability primitives, and isolated temporary
test storage. URL parsing is pinned to exactly `url 2.5.8`; the lock policy retains `idna 1.1.0`,
`idna_adapter 1.2.0`, and the Rust-1.88-compatible ICU4X 1.x graph. The current crate exposes an
optional `onnx-runtime` feature pinned to `ort 2.0.0-rc.13` and `ort-sys 2.0.0-rc.13`; default builds
still exclude that graph.

## Feature C: transport adapters

Owned paths:

- `crates/impossible-ocr-protocol/src/**`
- `crates/impossible-ocr-protocol/proto/**`
- `crates/impossible-ocr-protocol/build.rs`
- `crates/impossible-ocr-server/src/**`
- tests and fixtures beneath protocol and server

The prepared graph provides Axum HTTP, Prost/Tonic unary gRPC with a vendored `protoc`, base64,
Tokio stream and stdio support, and existing JSON support for a thin MCP adapter. Feature C owns the
protobuf schema and build script so those files can evolve atomically with generated API usage.

## Integration boundary

Feature A exposes validated canonical inputs/results and a backend contract. Feature B implements
bundle admission and a backend without owning wire formats. Feature C depends only on the public
domain/pipeline contracts and uses a deterministic fake backend in tests. No feature may add a
sibling-repository path dependency, network-on-startup behavior, native multimedia dependency, or
runtime model download outside the explicit installation operation.

## Dependency constraints

- Workspace dependencies are exact pins so a lockfile refresh cannot silently change the reviewed
  Wave 1 graph.
- Raster decoding is limited to PNG and JPEG; `image` default codecs remain disabled.
- Preprocessing and postprocessing must remain pure Rust. OpenCV, FFmpeg, native geometry libraries,
  and equivalent system-runtime dependencies are outside Wave 1.
- The current root `onnx-runtime` feature compiles only the exact ORT rc.13 graph with `std`,
  `load-dynamic`, and the reviewed `api-28` ABI selector; default builds exclude it. Runtime
  admission requires an explicit integrity-bound native library and must never search for or
  download one.
- HTTP, unary gRPC, and MCP exercise the same product contract through a deterministic fake backend;
  transport crates must not acquire model-store or algorithm ownership.
- The lockfile deliberately resolves `indexmap 2.11.4`/`hashbrown 0.15.5` and `flate2
  1.1.9`/`miniz_oxide 0.8.9`, eliminating duplicate `hashbrown` and `miniz_oxide` families without
  changing direct dependency pins. The only remaining duplicate families are `bitflags`,
  `getrandom`, and `syn`; cargo-deny has exact reviewed exceptions for `bitflags 1.3.2`, `getrandom
  0.2.17`, and `syn 3.0.5`, with no tree-wide exception. `scripts/verify-dependency-policy.ps1`
  fails on any new family, version drift, or exception drift and explicitly rejects reintroduction
  of an alpha `smallvec` duplicate family.
