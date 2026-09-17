# Model lifecycle

The built-in catalog admits one atomic bundle, `paddlex-ocr-3.7-max960`: the official
PaddlePaddle PP-OCRv5 mobile detector and English mobile recognizer. Both ONNX graphs and both
`inference.yml` files are pinned to immutable upstream revisions, exact byte lengths, SHA-256
digests, and Apache-2.0 provenance.

Opening `ModelStore` and normal server startup are offline operations. A caller must explicitly
invoke `install` to use the HTTPS downloader or `import` to copy named local files. The downloader
uses an exact host allowlist, bounded redirects, connect/idle/total timeouts, identity encoding,
streaming byte limits, and streaming SHA-256 verification. Errors and debug output omit source
URLs, redirect targets, query strings, and local paths.

The crate also contains a runtime-independent ONNX protobuf admission layer. It applies fixed byte,
field-count, recursion, node, tensor, attribute, value, and rank budgets before any native runtime
is loaded. Admission rejects external tensor data, non-standard operator domains, local functions,
training graphs, malformed or duplicate graph I/O, and any difference from an exact qualified
contract. The two checked-in contracts are qualified from the exact integrity-bound upstream
artifacts. Their artifact byte lengths and SHA-256 values are bound to the measured graph IR,
opsets, tensor signatures, and recursive operator multisets emitted deterministically by the bounded
parser. The `qualify-onnx` tool accepts only an explicit artifact path and curated role, verifies the
catalog byte length and SHA-256 before parsing, and emits no local path or host metadata. It never
searches for or downloads artifacts.

The optional `onnx-runtime` Cargo feature pins `ort` exactly and enables only `std`,
`load-dynamic`, and the reviewed `api-28` ABI selector. Default builds do not include ORT; neither
feature configuration downloads a native runtime or model weights. Native ONNX Runtime packaging
and session construction remain separate admission steps.

The total timeout is one absolute deadline measured from operation entry. Waiting for the
operation lock, every connection and redirect, body streaming (including slow-drip responses),
hashing, bounded local import reads, configuration validation, and promotion all spend the same
budget. Cancellation is checked between bounded reads and validation phases.

Installation holds a per-bundle process lock, writes into a same-directory staging tree, flushes
every artifact and the ownership marker, validates both verified YAML files, and promotes the
complete directory with a same-volume rename. An already verified bundle is unchanged. A corrupt
bundle is replaced only when its ownership marker matches the compiled catalog; unrelated data is
never claimed or deleted. Interrupted staging and corrupt owned bundles are handled only through
their exact catalog-derived paths, without searching the machine for models.
The prior owned directory is quarantined only for the duration of replacement, restored if
promotion fails, and removed after a successful repair.

The recognizer YAML must contain its ordered 436-entry dictionary, `use_space_char: true`, the
reviewed resize shapes, and the reviewed dynamic maximum. Verification derives a stable digest of
the ordered upstream dictionary. Runtime classes are blank index zero, the 436 entries, and the
appended literal space: 438 classes total.

The upstream detector YAML's `resize_long: 960` is verified and recorded as provenance, but it is
not silently adopted as the runtime preprocessing contract. The curated runtime profile remains
the explicit `paddlex-ocr-3.7-max960` maximum-side policy.

`status`, `verify`, and `delete` perform no network access. A `ModelLease` holds a shared filesystem
lock at the stable store root and an in-process reference until dropped; install, repair, and
deletion require that lock exclusively before download or replacement and retain it through
promotion. The stable lock cannot be renamed away with the bundle. Every existing component of
store, staging, and import paths is checked, so ancestor symbolic links, junctions, and other
Windows reparse points fail closed at import, verification, repair, and deletion boundaries.

Graph qualification does not instantiate or download ONNX Runtime. Runtime-binary integrity and
real inference remain separate release gates.
