use std::{
    fmt,
    io::{BufReader, Cursor, Read},
    panic::{AssertUnwindSafe, catch_unwind},
};

use image::{
    ColorType, DynamicImage, ImageDecoder, Limits,
    codecs::{jpeg::JpegDecoder, png::PngDecoder},
    metadata::Orientation,
};
use impossible_ocr_domain::{
    InputMetadata, OcrError, OcrErrorCode, RasterFormat, RasterInput, RasterLimits,
};

const PNG_SIGNATURE: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
const JPEG_SIGNATURE: &[u8; 3] = b"\xff\xd8\xff";
const JPEG_END: &[u8; 2] = b"\xff\xd9";

/// Policy for orientation metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrientationPolicy {
    /// Apply JPEG EXIF orientation before emitting RGB pixels.
    ApplyExif,
}

/// Policy for embedded color profiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColorProfilePolicy {
    /// Decode channel samples without an ICC color transform.
    Ignore,
}

/// Immutable metadata behavior of the v0.1 decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataPolicy {
    /// JPEG EXIF orientation behavior.
    pub orientation: OrientationPolicy,
    /// ICC profile behavior.
    pub color_profile: ColorProfilePolicy,
    /// Whether ancillary metadata is retained.
    pub retain_ancillary_metadata: bool,
}

impl Default for MetadataPolicy {
    fn default() -> Self {
        Self {
            orientation: OrientationPolicy::ApplyExif,
            color_profile: ColorProfilePolicy::Ignore,
            retain_ancillary_metadata: false,
        }
    }
}

/// Validated decoded RGB8 raster. Pixel bytes are redacted from `Debug`.
#[derive(Clone, PartialEq, Eq)]
pub struct DecodedRaster {
    width: u32,
    height: u32,
    rgb: Vec<u8>,
    source_format: RasterFormat,
    orientation_applied: bool,
}

impl DecodedRaster {
    /// Constructs a raster from exact RGB8 bytes.
    ///
    /// # Errors
    /// Returns `invalid_image` for zero dimensions or a mismatched byte length.
    pub fn from_rgb8(width: u32, height: u32, rgb: Vec<u8>) -> Result<Self, OcrError> {
        let expected = rgb_len(width, height)?;
        if width == 0 || height == 0 || rgb.len() != expected {
            return Err(invalid_image());
        }
        Ok(Self {
            width,
            height,
            rgb,
            source_format: RasterFormat::Png,
            orientation_applied: false,
        })
    }

    /// Pixel width after orientation is applied.
    #[must_use]
    pub const fn width(&self) -> u32 {
        self.width
    }

    /// Pixel height after orientation is applied.
    #[must_use]
    pub const fn height(&self) -> u32 {
        self.height
    }

    /// Contiguous RGB8 bytes in row-major order.
    #[must_use]
    pub fn rgb(&self) -> &[u8] {
        &self.rgb
    }

    /// Source encoding selected by magic sniffing.
    #[must_use]
    pub const fn source_format(&self) -> RasterFormat {
        self.source_format
    }

    /// Whether a non-identity EXIF orientation was applied.
    #[must_use]
    pub const fn orientation_applied(&self) -> bool {
        self.orientation_applied
    }
}

impl fmt::Debug for DecodedRaster {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodedRaster")
            .field("width", &self.width)
            .field("height", &self.height)
            .field("rgb", &"[REDACTED]")
            .field("source_format", &self.source_format)
            .field("orientation_applied", &self.orientation_applied)
            .finish()
    }
}

/// Bounded PNG/JPEG decoder with fail-closed metadata handling.
#[derive(Debug, Clone, Copy)]
pub struct RasterDecoder {
    limits: RasterLimits,
    metadata_policy: MetadataPolicy,
}

impl RasterDecoder {
    /// Creates a decoder with validated limits and the fixed v0.1 metadata policy.
    #[must_use]
    pub const fn new(limits: RasterLimits) -> Self {
        Self {
            limits,
            metadata_policy: MetadataPolicy {
                orientation: OrientationPolicy::ApplyExif,
                color_profile: ColorProfilePolicy::Ignore,
                retain_ancillary_metadata: false,
            },
        }
    }

    /// Returns the decoder's explicit metadata policy.
    #[must_use]
    pub const fn metadata_policy(self) -> MetadataPolicy {
        self.metadata_policy
    }

    /// Reads an encoded stream incrementally, rejecting it before retaining bytes beyond the cap.
    ///
    /// # Errors
    /// Returns a sanitized error for an I/O failure, limit violation, declaration mismatch, or
    /// invalid raster.
    pub fn decode_stream(
        &self,
        mut reader: impl Read,
        metadata: InputMetadata,
    ) -> Result<DecodedRaster, OcrError> {
        metadata.validate_with(self.limits)?;
        let initial_capacity = usize::try_from(metadata.encoded_bytes)
            .ok()
            .map_or(64 * 1024, |value| value.min(64 * 1024));
        let mut encoded = Vec::new();
        encoded
            .try_reserve_exact(initial_capacity)
            .map_err(|_| invalid_image())?;
        let mut chunk = [0_u8; 8192];
        let probe_limit = metadata
            .encoded_bytes
            .checked_add(1)
            .ok_or_else(invalid_image)?;
        loop {
            let retained = u64::try_from(encoded.len()).map_err(|_| invalid_image())?;
            let remaining = probe_limit
                .checked_sub(retained)
                .ok_or_else(invalid_image)?;
            if remaining == 0 {
                return Err(invalid_image());
            }
            let chunk_len = u64::try_from(chunk.len()).map_err(|_| invalid_image())?;
            let read_len =
                usize::try_from(remaining.min(chunk_len)).map_err(|_| invalid_image())?;
            let count = reader
                .read(&mut chunk[..read_len])
                .map_err(|_| invalid_image())?;
            if count == 0 {
                break;
            }
            let next_len = encoded.len().checked_add(count).ok_or_else(invalid_image)?;
            if u64::try_from(next_len).map_err(|_| invalid_image())? > metadata.encoded_bytes {
                return Err(invalid_image());
            }
            encoded.try_reserve(count).map_err(|_| invalid_image())?;
            encoded.extend_from_slice(&chunk[..count]);
        }
        let input = RasterInput::new(encoded, metadata)?;
        self.decode(&input)
    }

    /// Decodes one already-bounded raster with panic containment.
    ///
    /// # Errors
    /// Returns only sanitized `invalid_request` or `invalid_image` errors.
    pub fn decode(&self, input: &RasterInput) -> Result<DecodedRaster, OcrError> {
        input.metadata().validate_with(self.limits)?;
        catch_unwind(AssertUnwindSafe(|| self.decode_inner(input))).map_err(|_| invalid_image())?
    }

    fn decode_inner(&self, input: &RasterInput) -> Result<DecodedRaster, OcrError> {
        let format = sniff_format(input.bytes()).ok_or_else(invalid_image)?;
        if format != input.metadata().format {
            return Err(invalid_image());
        }
        match format {
            RasterFormat::Png => self.decode_png(input),
            RasterFormat::Jpeg => self.decode_jpeg(input),
        }
    }

    fn decode_png(&self, input: &RasterInput) -> Result<DecodedRaster, OcrError> {
        let cursor = Cursor::new(input.bytes());
        let reader = BufReader::new(cursor);
        let decoder = PngDecoder::with_limits(reader, image_limits(self.limits))
            .map_err(|_| invalid_image())?;
        if decoder.is_apng().map_err(|_| invalid_image())? {
            return Err(invalid_image());
        }
        validate_header(&decoder, input.metadata(), self.limits)?;
        let dynamic = DynamicImage::from_decoder(decoder).map_err(|_| invalid_image())?;
        finish_decode(
            dynamic,
            RasterFormat::Png,
            Orientation::NoTransforms,
            self.limits,
        )
    }

    fn decode_jpeg(&self, input: &RasterInput) -> Result<DecodedRaster, OcrError> {
        if !input.bytes().ends_with(JPEG_END) {
            return Err(invalid_image());
        }
        let cursor = Cursor::new(input.bytes());
        let mut decoder = JpegDecoder::new(BufReader::new(cursor)).map_err(|_| invalid_image())?;
        decoder
            .set_limits(image_limits(self.limits))
            .map_err(|_| invalid_image())?;
        validate_header(&decoder, input.metadata(), self.limits)?;
        let orientation = decoder.orientation().map_err(|_| invalid_image())?;
        let dynamic = DynamicImage::from_decoder(decoder).map_err(|_| invalid_image())?;
        finish_decode(dynamic, RasterFormat::Jpeg, orientation, self.limits)
    }
}

impl Default for RasterDecoder {
    fn default() -> Self {
        Self::new(RasterLimits::default())
    }
}

fn sniff_format(bytes: &[u8]) -> Option<RasterFormat> {
    if bytes.starts_with(PNG_SIGNATURE) {
        Some(RasterFormat::Png)
    } else if bytes.starts_with(JPEG_SIGNATURE) {
        Some(RasterFormat::Jpeg)
    } else {
        None
    }
}

fn image_limits(limits: RasterLimits) -> Limits {
    let mut image_limits = Limits::default();
    image_limits.max_image_width = Some(limits.max_width());
    image_limits.max_image_height = Some(limits.max_height());
    image_limits.max_alloc = Some(limits.max_decoded_bytes());
    image_limits
}

fn validate_header(
    decoder: &impl ImageDecoder,
    metadata: InputMetadata,
    limits: RasterLimits,
) -> Result<(), OcrError> {
    let (width, height) = decoder.dimensions();
    if width != metadata.width || height != metadata.height {
        return Err(invalid_image());
    }
    validate_decoded_shape(
        width,
        height,
        decoder.total_bytes(),
        decoder.color_type(),
        limits,
    )
}

fn validate_decoded_shape(
    width: u32,
    height: u32,
    decoder_bytes: u64,
    color: ColorType,
    limits: RasterLimits,
) -> Result<(), OcrError> {
    if width == 0
        || height == 0
        || width > limits.max_width()
        || height > limits.max_height()
        || !matches!(
            color,
            ColorType::L8 | ColorType::La8 | ColorType::Rgb8 | ColorType::Rgba8
        )
    {
        return Err(invalid_image());
    }
    let pixels = u64::from(width)
        .checked_mul(u64::from(height))
        .ok_or_else(invalid_image)?;
    let rgb_bytes = pixels.checked_mul(3).ok_or_else(invalid_image)?;
    if pixels > limits.max_pixels()
        || rgb_bytes > limits.max_decoded_bytes()
        || decoder_bytes > limits.max_decoded_bytes()
    {
        return Err(invalid_image());
    }
    Ok(())
}

fn finish_decode(
    mut image: DynamicImage,
    format: RasterFormat,
    orientation: Orientation,
    limits: RasterLimits,
) -> Result<DecodedRaster, OcrError> {
    let orientation_applied = orientation != Orientation::NoTransforms;
    image.apply_orientation(orientation);
    let width = image.width();
    let height = image.height();
    let rgb_len = rgb_len(width, height)?;
    validate_decoded_shape(
        width,
        height,
        u64::try_from(rgb_len).map_err(|_| invalid_image())?,
        ColorType::Rgb8,
        limits,
    )?;
    if u64::try_from(rgb_len).map_err(|_| invalid_image())? > limits.max_decoded_bytes() {
        return Err(invalid_image());
    }
    let rgb = into_rgb_on_white(image, rgb_len)?;
    Ok(DecodedRaster {
        width,
        height,
        rgb,
        source_format: format,
        orientation_applied,
    })
}

fn into_rgb_on_white(image: DynamicImage, expected: usize) -> Result<Vec<u8>, OcrError> {
    if let DynamicImage::ImageRgb8(image) = image {
        let raw = image.into_raw();
        return (raw.len() == expected)
            .then_some(raw)
            .ok_or_else(invalid_image);
    }
    let rgba = image.into_rgba8();
    let mut rgb = Vec::new();
    rgb.try_reserve_exact(expected)
        .map_err(|_| invalid_image())?;
    for pixel in rgba.pixels() {
        let [red, green, blue, alpha] = pixel.0;
        rgb.push(composite_on_white(red, alpha)?);
        rgb.push(composite_on_white(green, alpha)?);
        rgb.push(composite_on_white(blue, alpha)?);
    }
    (rgb.len() == expected)
        .then_some(rgb)
        .ok_or_else(invalid_image)
}

fn composite_on_white(channel: u8, alpha: u8) -> Result<u8, OcrError> {
    let alpha = u16::from(alpha);
    let value = (u16::from(channel) * alpha + 255 * (255 - alpha) + 127) / 255;
    u8::try_from(value).map_err(|_| invalid_image())
}

fn rgb_len(width: u32, height: u32) -> Result<usize, OcrError> {
    let bytes = u64::from(width)
        .checked_mul(u64::from(height))
        .and_then(|pixels| pixels.checked_mul(3))
        .ok_or_else(invalid_image)?;
    usize::try_from(bytes).map_err(|_| invalid_image())
}

const fn invalid_image() -> OcrError {
    OcrError::for_code(OcrErrorCode::InvalidImage)
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;

    use image::{
        ColorType, ImageEncoder,
        codecs::{jpeg::JpegEncoder, png::PngEncoder},
    };
    use impossible_ocr_domain::{
        InputMetadata, OcrErrorCode, RasterFormat, RasterInput, RasterLimits,
    };

    use super::{ColorProfilePolicy, OrientationPolicy, RasterDecoder};

    fn png(width: u32, height: u32, rgba: &[u8]) -> Result<Vec<u8>, image::ImageError> {
        let mut bytes = Vec::new();
        PngEncoder::new(&mut bytes).write_image(rgba, width, height, ColorType::Rgba8.into())?;
        Ok(bytes)
    }

    fn jpeg(width: u32, height: u32, rgb: &[u8]) -> Result<Vec<u8>, image::ImageError> {
        let mut bytes = Vec::new();
        JpegEncoder::new_with_quality(&mut bytes, 90).write_image(
            rgb,
            width,
            height,
            ColorType::Rgb8.into(),
        )?;
        Ok(bytes)
    }

    fn input(
        bytes: Vec<u8>,
        format: RasterFormat,
        width: u32,
        height: u32,
    ) -> Result<RasterInput, impossible_ocr_domain::OcrError> {
        let encoded_bytes = u64::try_from(bytes.len())
            .map_err(|_| impossible_ocr_domain::OcrError::for_code(OcrErrorCode::InvalidRequest))?;
        RasterInput::new(
            bytes,
            InputMetadata {
                format,
                encoded_bytes,
                width,
                height,
            },
        )
    }

    #[test]
    fn png_alpha_is_flattened_on_white_and_bytes_are_redacted()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = png(2, 1, &[255, 0, 0, 255, 0, 0, 0, 0])?;
        let decoded = RasterDecoder::default().decode(&input(bytes, RasterFormat::Png, 2, 1)?)?;
        assert_eq!(decoded.rgb(), &[255, 0, 0, 255, 255, 255]);
        assert!(!format!("{decoded:?}").contains("255, 0, 0"));
        Ok(())
    }

    #[test]
    fn jpeg_and_png_magic_must_match_the_declaration() -> Result<(), Box<dyn std::error::Error>> {
        let png_bytes = png(1, 1, &[1, 2, 3, 255])?;
        assert_eq!(
            RasterDecoder::default()
                .decode(&input(png_bytes, RasterFormat::Jpeg, 1, 1)?)
                .map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::InvalidImage)
        );
        let jpeg_bytes = jpeg(1, 1, &[1, 2, 3])?;
        assert!(
            RasterDecoder::default()
                .decode(&input(jpeg_bytes, RasterFormat::Jpeg, 1, 1)?)
                .is_ok()
        );
        Ok(())
    }

    #[test]
    fn dimensions_truncation_and_malformed_bytes_fail_closed()
    -> Result<(), Box<dyn std::error::Error>> {
        let bytes = png(1, 1, &[1, 2, 3, 255])?;
        assert!(
            RasterDecoder::default()
                .decode(&input(bytes.clone(), RasterFormat::Png, 2, 1)?)
                .is_err()
        );
        let truncated = bytes[..bytes.len() / 2].to_vec();
        assert!(
            RasterDecoder::default()
                .decode(&input(truncated, RasterFormat::Png, 1, 1)?)
                .is_err()
        );
        let garbage = Vec::from(*b"not-an-image");
        assert!(
            RasterDecoder::default()
                .decode(&input(garbage, RasterFormat::Png, 1, 1)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn unsupported_16_bit_png_and_decoded_byte_overrun_are_rejected()
    -> Result<(), Box<dyn std::error::Error>> {
        let mut encoded_16_bit = Vec::new();
        PngEncoder::new(&mut encoded_16_bit).write_image(&[0, 1], 1, 1, ColorType::L16.into())?;
        assert!(
            RasterDecoder::default()
                .decode(&input(encoded_16_bit, RasterFormat::Png, 1, 1)?)
                .is_err()
        );

        let rgba = png(1, 1, &[1, 2, 3, 4])?;
        let limits = RasterLimits::new(1024, 1, 1, 1, 3)?;
        assert!(
            RasterDecoder::new(limits)
                .decode(&input(rgba, RasterFormat::Png, 1, 1)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn jpeg_exif_orientation_is_applied_after_header_agreement()
    -> Result<(), Box<dyn std::error::Error>> {
        let encoded = jpeg(2, 1, &[255, 0, 0, 0, 0, 255])?;
        let oriented = insert_exif_orientation(encoded, 6)?;
        let decoded =
            RasterDecoder::default().decode(&input(oriented, RasterFormat::Jpeg, 2, 1)?)?;
        assert_eq!((decoded.width(), decoded.height()), (1, 2));
        assert!(decoded.orientation_applied());
        Ok(())
    }

    #[test]
    fn streamed_input_stops_at_the_configured_encoded_bound()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = RasterLimits::new(8, 8, 8, 64, 192)?;
        let decoder = RasterDecoder::new(limits);
        let metadata = InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes: 8,
            width: 1,
            height: 1,
        };
        let error = decoder.decode_stream(Cursor::new([0_u8; 9]), metadata);
        assert_eq!(
            error.map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::InvalidImage)
        );
        Ok(())
    }

    #[test]
    fn streamed_input_reads_at_most_declared_length_plus_one()
    -> Result<(), Box<dyn std::error::Error>> {
        let limits = RasterLimits::new(32 * 1024, 8, 8, 64, 192)?;
        let decoder = RasterDecoder::new(limits);
        let metadata = InputMetadata {
            format: RasterFormat::Png,
            encoded_bytes: 8,
            width: 1,
            height: 1,
        };
        let mut input = Cursor::new(vec![0_u8; 32 * 1024]);
        assert!(decoder.decode_stream(&mut input, metadata).is_err());
        assert_eq!(input.position(), 9);
        Ok(())
    }

    #[test]
    fn orientation_is_revalidated_against_asymmetric_limits()
    -> Result<(), Box<dyn std::error::Error>> {
        let encoded = insert_exif_orientation(jpeg(2, 1, &[255, 0, 0, 0, 0, 255])?, 6)?;
        let limits = RasterLimits::new(u64::try_from(encoded.len())?, 2, 1, 2, 6)?;
        let candidate = input(encoded, RasterFormat::Jpeg, 2, 1)?;
        assert_eq!(
            RasterDecoder::new(limits)
                .decode(&candidate)
                .map_err(impossible_ocr_domain::OcrError::code),
            Err(OcrErrorCode::InvalidImage)
        );
        Ok(())
    }

    #[test]
    fn apng_control_chunk_is_rejected() -> Result<(), Box<dyn std::error::Error>> {
        let bytes = png(1, 1, &[1, 2, 3, 255])?;
        let animated = insert_actl(bytes)?;
        assert!(
            RasterDecoder::default()
                .decode(&input(animated, RasterFormat::Png, 1, 1)?)
                .is_err()
        );
        Ok(())
    }

    #[test]
    fn metadata_policy_is_explicit_and_non_retaining() {
        let policy = RasterDecoder::default().metadata_policy();
        assert_eq!(policy.orientation, OrientationPolicy::ApplyExif);
        assert_eq!(policy.color_profile, ColorProfilePolicy::Ignore);
        assert!(!policy.retain_ancillary_metadata);
    }

    #[test]
    fn malformed_short_inputs_never_escape_as_panics() -> Result<(), impossible_ocr_domain::OcrError>
    {
        for length in 1..64_u64 {
            let bytes = vec![
                0_u8;
                usize::try_from(length).map_err(|_| {
                    impossible_ocr_domain::OcrError::for_code(OcrErrorCode::InvalidRequest)
                })?
            ];
            let candidate = RasterInput::new(
                bytes,
                InputMetadata {
                    format: RasterFormat::Png,
                    encoded_bytes: length,
                    width: 1,
                    height: 1,
                },
            )?;
            assert!(RasterDecoder::default().decode(&candidate).is_err());
        }
        Ok(())
    }

    fn insert_actl(mut png: Vec<u8>) -> Result<Vec<u8>, &'static str> {
        const AFTER_IHDR: usize = 8 + 4 + 4 + 13 + 4;
        if png.len() < AFTER_IHDR {
            return Err("encoded PNG omitted IHDR");
        }
        let mut chunk = Vec::from([0, 0, 0, 8]);
        chunk.extend_from_slice(b"acTL");
        chunk.extend_from_slice(&1_u32.to_be_bytes());
        chunk.extend_from_slice(&0_u32.to_be_bytes());
        let crc = crc32(&chunk[4..]);
        chunk.extend_from_slice(&crc.to_be_bytes());
        png.splice(AFTER_IHDR..AFTER_IHDR, chunk);
        Ok(png)
    }

    fn insert_exif_orientation(
        mut jpeg: Vec<u8>,
        orientation: u16,
    ) -> Result<Vec<u8>, &'static str> {
        if !jpeg.starts_with(&[0xff, 0xd8]) {
            return Err("encoded JPEG omitted SOI");
        }
        let mut payload = Vec::from(*b"Exif\0\0");
        payload.extend_from_slice(b"MM\0*");
        payload.extend_from_slice(&8_u32.to_be_bytes());
        payload.extend_from_slice(&1_u16.to_be_bytes());
        payload.extend_from_slice(&0x0112_u16.to_be_bytes());
        payload.extend_from_slice(&3_u16.to_be_bytes());
        payload.extend_from_slice(&1_u32.to_be_bytes());
        payload.extend_from_slice(&orientation.to_be_bytes());
        payload.extend_from_slice(&0_u16.to_be_bytes());
        payload.extend_from_slice(&0_u32.to_be_bytes());
        let segment_len = u16::try_from(payload.len() + 2).map_err(|_| "EXIF too large")?;
        let mut segment = vec![0xff, 0xe1];
        segment.extend_from_slice(&segment_len.to_be_bytes());
        segment.extend_from_slice(&payload);
        jpeg.splice(2..2, segment);
        Ok(jpeg)
    }

    fn crc32(bytes: &[u8]) -> u32 {
        let mut crc = u32::MAX;
        for byte in bytes {
            crc ^= u32::from(*byte);
            for _ in 0..8 {
                let mask = 0_u32.wrapping_sub(crc & 1);
                crc = (crc >> 1) ^ (0xedb8_8320 & mask);
            }
        }
        !crc
    }

    #[test]
    fn rgb_constructor_rejects_wrong_length() {
        assert!(super::DecodedRaster::from_rgb8(2, 2, vec![0; 11]).is_err());
        assert!(super::DecodedRaster::from_rgb8(u32::MAX, u32::MAX, Vec::new()).is_err());
    }
}
