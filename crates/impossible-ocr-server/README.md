# Impossible OCR transport host

This crate exposes one canonical synchronous raster OCR operation over HTTP, unary gRPC, and MCP.
All transports accept only static PNG or JPEG input and share the same pipeline, limits, stable error
categories, request identifiers, readiness, and cancellation path.

HTTP listens on loopback by default and includes JSON/base64 single and batch operations plus a raw
single-raster endpoint. The raw endpoint requires matching `Content-Type`, `x-image-format`,
`x-image-width`, and `x-image-height` headers. Multipart is not enabled because the reviewed Axum
feature set does not include a multipart parser. PDF, jobs, and WebSockets are not advertised.

gRPC provides unary `Recognize`, `RecognizeBatch`, and `GetCapabilities`. MCP provides
`ocr_recognize`, `ocr_batch`, `ocr_capabilities`, and `ocr_models` over Streamable HTTP or
newline-delimited stdio. The checked-in OpenAPI document, protobuf source, and MCP JSON schemas are
the authoritative transport descriptions. JSON-RPC notifications, including
`notifications/initialized`, execute without a response; an explicit `id: null` remains a request
and receives a correlated response.

Network listeners are loopback-only. There is no public-bind or remote-authentication mode in v0.1.
Input bytes, filenames, recognized text, paths, and request metadata are never written to logs or
errors. Dropping a handler future cancels its backend future; deadline and shutdown cancellation use
the shared request context. Shutdown stops admission, cancels active contexts, and has a strict
configured bound.

HTTP uses a deadline-enforcing listener wrapper: every accepted connection has a maximum read
lifetime equal to the configured request timeout. This bounds incomplete headers, slow bodies, and
idle keep-alive connections; long-lived clients reconnect after that lifetime. Once headers are
decoded, admission, concurrency, and the same total request deadline wrap body extraction and route
execution. Raw, JSON/base64, batch, MCP, and gRPC use distinct advertised wire caps while sharing a
separate encoded-image cap and aggregate batch-image cap.

`--max-request-bytes` (or `IMPOSSIBLE_OCR_MAX_REQUEST_BYTES`) configures the maximum encoded
PNG/JPEG bytes in one image after base64 or protobuf decoding. It is not a universal wire-body cap:
the server derives separate raw, JSON/base64, batch, MCP, and gRPC wire limits from that image cap
and the fixed protocol limits.

Tonic applies a per-connection concurrency limit, load shedding, a service timeout, and one
service-wide protobuf decode cap before handler dispatch. Tonic does not expose different decoder
caps per method on one generated service, so unary and batch requests share the advertised gRPC wire
cap; handlers immediately validate the stricter single-image and aggregate-batch byte limits after
protobuf decoding. A future public-bind mode would still require an authenticated edge proxy and
transport-specific abuse controls.

## Offline ONNX startup

The default build exposes the complete transport shell but remains false-ready. The optional
`onnx-runtime` feature enables the curated CPU backend. All five backend inputs are atomic: model
store, runtime directory, runtime filename, exact native-library byte length, and lowercase SHA-256.
If any are absent or invalid, startup fails or deliberately retains the unavailable shell.

Startup verifies the already-installed curated bundle and graph contracts, admits the native
library into owned content-addressed storage, creates a bounded number of detector/recognizer
session pairs, and warms every lane before readiness becomes true. It never scans the host, searches
`PATH`, or downloads artifacts. `--runtime-lanes` cannot exceed request concurrency, and lanes times
the larger per-session thread setting cannot exceed `--runtime-cpu-budget`.

Use the `import-model-bundle` binary to import the four exact, already-downloaded PaddlePaddle model
and configuration files into a chosen store. The import is offline, verifies every pinned size and
SHA-256, validates the configuration and dictionary, and promotes the complete bundle atomically.
It never accepts a source URL or scans for model files.

The server reports the curated model identifier only while the warmed backend is ready. A warm-up
failure leaves readiness false. Ctrl-C starts the same bounded shutdown and cancellation flow for
HTTP, gRPC, and MCP stdio.
