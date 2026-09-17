# Backlog and cross-project lessons

This page records optional follow-up work discovered while building Impossible OCR. None of these
items blocks the current v0.1 contract, carries a delivery date, or promises inclusion in a future
release. Items should be reconsidered against measured user needs before implementation.

## OCR hardening

- Expand adversarial raster coverage for rotation, perspective, low contrast, dense layouts,
  extreme aspect ratios, malformed metadata, and decoder edge cases without retaining user images.
- Add longer cancellation, concurrency, shutdown, and bounded-memory soak exercises around real
  inference sessions.
- Extend fuzzing across raster admission, ONNX graph admission, model lifecycle manifests, and the
  HTTP, gRPC, and MCP boundaries.
- Improve privacy-safe diagnostics and operator runbooks without exposing paths, recognized text,
  tensor values, timings tied to a host, or machine specifications.

## Neutral server foundation backfeed

- Generalize integrity-bound native-runtime admission and content-addressed artifact installation
  without moving OCR-specific tensor or postprocessing behavior into the shared foundation.
- Reuse the closed-schema qualification-attestation pattern: deterministic probes, source and
  artifact bindings, strict mutation tests, and explicit platform scope.
- Carry forward the separation between packaging qualification, runtime inference qualification,
  readiness, and ordinary liveness.
- Review transport admission, deadlines, cancellation, metrics, and privacy-safe error patterns for
  modality-neutral extraction only after OCR release behavior is stable.

## Platform qualification

- Run the pinned production harness natively on Linux x86-64 before calling Linux inference
  qualified; Windows execution is not substitute evidence.
- Consider additional operating systems and architectures only with exact runtime manifests,
  native execution evidence, packaging checks, and the same privacy constraints.

## Models and performance

- Evaluate multilingual, document-layout, orientation, handwriting, and alternative OCR bundles as
  separate curated profiles rather than silently changing v0.1 behavior.
- Measure batching, thread budgets, memory ceilings, and accelerator providers with reproducible
  workloads that publish no host fingerprint or input content.
- Consider GPU providers and additional CPU targets only when their deployment and failure modes can
  remain explicit, bounded, offline at startup, and independently qualified.

## Future Voice and inference convergence

- Reuse the neutral control plane, lifecycle, attestation, health, metrics, and transport contracts
  where the semantics truly match across OCR, Voice, and other local inference services.
- Keep modality-specific concerns separate: OCR geometry and reading order, Voice streaming and
  resampling, and model-specific tensor pipelines should not be forced into one abstraction.
- Feed proven lessons back into the shared skeleton before using it as the base for further services;
  this backlog does not begin that implementation work.
