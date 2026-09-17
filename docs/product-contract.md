# Raster OCR v0.1 contract

Version 0.1 accepts exactly one static PNG or JPEG image per synchronous request. It returns a
single page containing deterministic blocks, lines, words, quadrilateral polygons, UTF-8 text, and
finite confidence values in the inclusive range 0 through 1.

The initial qualified model scope is English text, ASCII digits, punctuation, and spaces. PDF,
multi-page images, animation, layout extraction, tables, handwriting guarantees, orientation
classification, arbitrary model upload, and remote model download are outside the contract.

Encoded input size, dimensions, decoded pixels, concurrency, queue depth, request lifetime, and
shutdown lifetime are bounded. MIME labels are advisory: a future decoder must verify magic bytes.
Results must use page index zero and top-to-bottom, then left-to-right ordering. Coordinates must be
finite and lie within the page bounds.

No image bytes, extracted text, metadata, model paths, or private diagnostics may be written to
normal logs or public errors. Startup and inference are offline. Readiness is false unless an entire
detector-plus-English-recognizer bundle has passed integrity checks and both sessions are warm.

## Raster admission and metadata

The decoder identifies PNG and JPEG by magic bytes and requires the declared format and dimensions
to match the encoded headers. Encoded bytes are accumulated incrementally under the configured cap;
width, height, pixel count, decoder allocation, and final RGB8 byte count are checked before the
corresponding large allocation. Zero-sized, truncated, malformed, unsupported-color, and animated
PNG inputs fail with a sanitized `invalid_image` error. Decoder panics are contained at the product
boundary and are never allowed to unwind through a transport.

All successful inputs become deterministic RGB8. Alpha is composited on white with integer rounding.
Non-identity JPEG EXIF orientation is applied before preprocessing. ICC profiles and all other
ancillary metadata are ignored and never retained or returned; no color-profile transform is
performed. PNG textual chunks are ignored by the pinned decoder. These behaviors are part of v0.1,
not runtime options.

## Pinned pure transforms

The detector transform is named `paddlex-ocr-3.7-max960`. Images whose height plus width is below 64
are first black-padded at bottom/right to at least 32 by 32. Images with a side above 960 are reduced
proportionally; preliminary dimensions are truncated and then rounded to the nearest multiple of 32
using Python ties-to-even semantics, with a minimum side of 32. The entire image is resized to that
exact tensor shape using bilinear interpolation, so stride rounding can slightly enlarge or shrink a
side even when max-side scaling itself does not upscale. RGB8 becomes BGR, scales by `1/255`, uses BGR means
`[0.485, 0.456, 0.406]` and standard deviations `[0.229, 0.224, 0.225]`, and is emitted as NCHW.
Original dimensions, post-tiny-padding dimensions, scale factors, and tensor dimensions are retained
for later box projection.

Detector postprocessing is the pinned PP-OCRv5 DB `quad`/`fast` profile. Model probabilities must
be finite and lie in the inclusive range zero through one. Foreground uses the strict `> 0.3`
threshold with no dilation. Deterministic raster-order, eight-neighbor contours are capped at 1,000
candidates before filtering. A candidate's minimum-area rectangle must have a short side of at least
3 pixels; its score is the mean original probability under the integer-filled rectangle and must be
at least 0.6. The rectangle is expanded by `area * 1.5 / perimeter` on every local side and its new
short side must be at least 5 pixels. Corners are projected to the decoded image using ties-to-even
rounding after division by the recorded detector scale, clipped to the last valid source pixel, and
returned in top-left, top-right, bottom-right, bottom-left order. Cancellation and deadlines are
checked throughout contour and candidate processing.

Recognizer crops are stably sorted by aspect ratio and restored to caller order after inference.
Each crop is resized to height 48 while preserving aspect ratio, capped at width 3200, converted to
BGR, normalized as `value / 127.5 - 1`, and right-padded with exact zero. The shared dynamic width is
at least 320 and no larger than 3200.

The greedy CTC decoder receives the exact model dictionary. Class zero is blank. Identical adjacent
classes collapse unless separated by blank; an optional space is appended exactly once to the
dictionary. Ties choose the lowest class index. Confidence is the arithmetic mean of probabilities
for emitted classes and is zero for empty output. Shape/class mismatches and non-finite or out-of-range
probabilities fail closed as sanitized internal model-output errors.

The transport-independent PP-OCR engine runs blocking decode, preprocessing, inference, and
postprocessing work outside the async executor. It checks cancellation before deadline between every
stage, caps aggregate temporary crop pixels at the product pixel limit, and invokes recognition in
chunks of at most eight crops. Empty detection skips crop and recognizer work. Regions retain the
canonical Paddle reading order across chunk boundaries; empty or below-threshold recognition is
filtered without reordering survivors. Each retained region produces one block and one line. When
`include_words` is enabled, the line also contains exactly one word with the same text, confidence,
and region polygon; the engine never fabricates sub-word geometry.
