# Native ONNX Runtime packaging

Impossible OCR uses the CPU build of Microsoft ONNX Runtime 1.28.0 through the optional `ort
2.0.0-rc.13` dependency. Native runtime acquisition is an explicit administrator operation. The
server never downloads a runtime during startup or request handling.

The versioned manifest in
`crates/impossible-ocr-onnx/runtime/onnxruntime-1.28.0.json` pins Microsoft's release commit, release
and asset identifiers, exact archive lengths, and GitHub-published SHA-256 digests for Windows x64
and Linux x64. The GitHub release is not marked immutable, so a matching version or URL is never
sufficient: length and digest must both match.

`schemas/onnxruntime-runtime-manifest.schema.json` defines the versioned manifest shape, including
safe relative members, archive bounds, per-file integrity records, exact-member rules, and narrowly
reviewed ignored links. The packaging test validates the production manifest against that schema
before exercising installer behavior.

## Qualification state

Archive-level and required-file values are complete for the official Windows x64 and Linux x64
archives. Each platform's five promoted files is recorded with its exact length and SHA-256.
Windows `Install` and `Verify` pass against the content-addressed installation. Windows inference
is release-qualified only when
`attestations/runtime/windows-x86_64-onnxruntime-1.28.0.json` exists and passes the strict verifier.
That attestation is generated only by a fresh real-model run through the recorder below. The Linux
archive was inspected without executing its binaries;
its archive digest, complete member allowlist, required files, redistribution notices, and two
convenience symbolic links were reviewed and pinned. Linux runtime execution remains a
platform-native release qualification step rather than something inferred from Windows.

1. Fetch the selected official archive outside server startup and independently verify its exact
   archive length and published SHA-256.
2. Run `pwsh scripts/install-onnxruntime.ps1 -Action Inspect -Platform <id> -ArchivePath <path>`.
   Inspection never downloads or installs. It validates the archive and prints the derived records.
3. Review archive members, PE imports or ELF `DT_NEEDED`, architecture, runtime version, C API 28,
   CPU-only provider behavior, Microsoft notices, and warm-up/golden inference in an isolated host.
4. Put the reviewed per-file lengths and SHA-256 values into the manifest. Review the manifest diff.
5. Run the test script and both offline `Verify` and explicit `Install` smoke checks.

After the per-file values are reviewed and the runtime plus curated model bundle are installed,
record the opt-in Windows inference qualification. The recorder invokes the real harness offline,
validates its exact probe matrix, and atomically writes the only accepted attestation path:

```text
pwsh ./scripts/record-runtime-qualification-attestation.ps1 \
  -ModelStore <absolute-model-store> \
  -RuntimeDirectory <absolute-content-addressed-Windows-runtime-directory>
```

The recorder persists only fixed public artifact identities, canonical source and contract digests,
and the harness's deterministic boolean probe report—never paths, host specifications, timings,
tensors, or recognized text. `scripts/verify-runtime-qualification-attestation.ps1` validates the
closed schema, rejects path-like strings and unknown fields, recomputes every binding, and requires
the exact warm-up, detector, recognizer, expected-admission, and generated-PNG results. The
native-free mutant suite exercises those rejection paths without creating or modifying an
attestation.

Windows is both packaging- and inference-qualified only while that verifier passes. Linux is
packaging-qualified but inference-unqualified: its binaries were not executed on Windows, and a
Windows run cannot stand in for platform-native Linux evidence. Both runtime-manifest entries have
complete required-file hashes; package integrity and inference qualification are deliberately
separate claims.

Do not copy values between the GitHub archives and NuGet: separately published packages need not be
bit-identical.

## Installed layout

The installer promotes only the main runtime, the provider sidecar, `LICENSE`,
`ThirdPartyNotices.txt`, and `Privacy.md`. They live below
`<runtime-root>/onnxruntime/1.28.0/<platform>/<archive-sha256>/`. Files are streamed into a private
staging directory, flushed, verified, made read-only, and atomically renamed. The parent directory
is flushed after the rename on Windows and Linux. Runtime loading must use the absolute path inside
this content-addressed directory and revalidate the recorded file identity and digest.

The source archive is validated before extraction. Absolute paths, `..`, backslashes, alternate
data streams, duplicate or case-colliding names, links, reparse metadata, devices, unexpected
members, unexpected executables, excessive members, and excessive expanded bytes are rejected.
Only allowlisted regular members are copied; archive extraction helpers are not used.

Linux release archives commonly carry convenience symbolic links. Links are rejected by default.
For this exact archive, the manifest identifies the two reviewed `libonnxruntime.so` convenience
links and their literal relative targets. The inspector verifies and ignores those exact entries;
it never extracts or follows them. Any additional link, changed target, special member, or member
outside the reviewed allowlist fails inspection.

## Network boundary

`Install` downloads only when `-AllowDownload` is explicitly present and no `-ArchivePath` was
provided. Every hop must be HTTPS and use a manifest-allowlisted host. DNS answers must be public.
Redirects, response length, streamed byte count, idle-read duration, and archive digest are bounded.
The installer never accepts credentials, custom headers, proxy URLs, or a caller-supplied source URL.

## Redistribution

ONNX Runtime is MIT-licensed. Redistributed packages must include Microsoft's exact `LICENSE`,
`ThirdPartyNotices.txt`, and `Privacy.md` from the verified artifact. Third-party notices are kept
verbatim and are not replaced by a synthesized license list. Release validation must fail if any
notice is absent or has a different recorded digest.

Official provenance:

- https://github.com/microsoft/onnxruntime/releases/tag/v1.28.0
- https://api.github.com/repos/microsoft/onnxruntime/releases/tags/v1.28.0
- https://github.com/microsoft/onnxruntime/blob/v1.28.0/LICENSE
- https://github.com/microsoft/onnxruntime/blob/v1.28.0/ThirdPartyNotices.txt
