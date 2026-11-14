use std::collections::HashSet;
use std::io::Write;

use flate2::{write::ZlibEncoder, Compression};
use image::codecs::avif::AvifEncoder;
use image::codecs::jpeg::JpegEncoder;
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::codecs::webp::WebPEncoder;
use image::{
    imageops::FilterType, DynamicImage, ExtendedColorType, GrayImage, ImageEncoder, ImageFormat,
    RgbImage,
};
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, Stream};
use thiserror::Error;

/// Maximum uncompressed stream size (256 MiB) per image layer to guard against corrupt streams.
const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;
/// Minimum pixel dimension; images strictly smaller in width or height are skipped (icons/artifacts).
const MIN_IMAGE_DIMENSION: u32 = 50;
/// Binary MiB divisor used consistently for MB readouts and target-size limits.
pub const BYTES_PER_MB: f64 = 1024.0 * 1024.0;
/// Pixel cap used when sample-compressing a single representative image during PDF load.
const MAX_CALIBRATION_PIXELS: u64 = 1_000_000;
/// Ratio of grayscale to RGB bits-per-pixel for JPEG and Flate streams.
const GRAY_TO_RGB_JPEG_RATIO: f64 = 0.72;
const GRAY_TO_RGB_FLATE_RATIO: f64 = 0.42;
/// Fallback bits-per-pixel defaults if no image could be sample-calibrated on load.
const DEFAULT_RGB_JPEG_Q50_BPP: f64 = 1.05;
const DEFAULT_RGB_FLATE_BPP: f64 = 8.50;

pub fn bytes_to_mb(bytes: usize) -> f64 {
    (bytes as f64) / BYTES_PER_MB
}

pub fn mb_to_bytes(mb: f64) -> usize {
    if mb <= 0.0 || !mb.is_finite() {
        0
    } else {
        (mb * BYTES_PER_MB).round() as usize
    }
}

#[derive(Debug, Error)]
pub enum PdfError {
    #[error("Failed to parse PDF: {0}")]
    Parse(String),
    #[error("This PDF is password-protected / encrypted and cannot be unlocked automatically.")]
    Encrypted,
    #[error("Failed to serialize compressed PDF: {0}")]
    Save(String),
    #[error("Failed to encode image stream: {0}")]
    ImageEncode(String),
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ImageFormatChoice {
    Jpeg,
    Png,
    Webp,
    Avif,
}

impl ImageFormatChoice {
    pub const ALL: [Self; 4] = [Self::Jpeg, Self::Png, Self::Webp, Self::Avif];

    pub fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG (/DCTDecode — Recommended for photos)",
            Self::Png => "PNG (/FlateDecode — Lossless)",
            Self::Webp => "WebP (Transcoded -> PDF /FlateDecode)",
            Self::Avif => "AVIF (Transcoded -> PDF /DCTDecode)",
        }
    }

    pub fn short_name(self) -> &'static str {
        match self {
            Self::Jpeg => "JPEG",
            Self::Png => "PNG (Lossless)",
            Self::Webp => "WebP",
            Self::Avif => "AVIF",
        }
    }

    pub fn uses_quality(self) -> bool {
        matches!(self, Self::Jpeg | Self::Avif)
    }

    pub fn compatibility_note(self) -> Option<&'static str> {
        match self {
            Self::Jpeg | Self::Png => None,
            Self::Webp => Some(
                "Note: Standard PDF readers (Chrome, Adobe Acrobat, Apple Preview) do not support native WebP streams per ISO 32000. Images are processed via pure-Rust WebP and written as lossless PDF /FlateDecode streams so every viewer opens the PDF without errors.",
            ),
            Self::Avif => Some(
                "Note: Standard PDF readers (Chrome, Adobe Acrobat, Apple Preview) do not support native AVIF streams per ISO 32000. Images are encoded via pure-Rust AVIF (ravif) for size estimation and written as universally compatible PDF /DCTDecode (JPEG) streams using your Quality setting.",
            ),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResizeFilterChoice {
    Lanczos3,
    CatmullRom,
}

impl ResizeFilterChoice {
    pub const ALL: [Self; 2] = [Self::Lanczos3, Self::CatmullRom];

    pub fn label(self) -> &'static str {
        match self {
            Self::Lanczos3 => "Lanczos3 (Sharpest)",
            Self::CatmullRom => "Catmull-Rom (Fast & crisp)",
        }
    }

    fn to_image_filter(self) -> FilterType {
        match self {
            Self::Lanczos3 => FilterType::Lanczos3,
            Self::CatmullRom => FilterType::CatmullRom,
        }
    }
}

#[derive(Debug, Clone)]
pub struct CompressionOptions {
    pub format: ImageFormatChoice,
    pub quality: u8,
    pub scale: f32,
    pub resize_filter: ResizeFilterChoice,
}

impl Default for CompressionOptions {
    fn default() -> Self {
        Self {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PdfSummary {
    pub page_count: usize,
    pub image_count: usize,
    pub compressible_image_count: usize,
    pub file_bytes: usize,
    pub non_image_overhead_bytes: usize,
    pub compressible_rgb_pixels: u64,
    pub compressible_gray_pixels: u64,
    pub orig_dct_image_count: usize,
    pub orig_dct_bytes: usize,
    pub orig_dct_rgb_pixels: u64,
    pub orig_dct_gray_pixels: u64,
    pub sample_rgb_jpeg_q50_bpp: f64,
    pub sample_rgb_flate_bpp: f64,
}

impl PdfSummary {
    /// Estimates the compressed PDF size in bytes using O(1) pure arithmetic from load-time
    /// inspection metrics and optional single-image calibration, keeping UI slider drags instant.
    pub fn estimate_compressed_bytes(&self, options: &CompressionOptions) -> usize {
        self.estimate_compressed_bytes_impl(options, true)
    }

    fn estimate_compressed_bytes_impl(
        &self,
        options: &CompressionOptions,
        apply_orig_dct_cap: bool,
    ) -> usize {
        if self.compressible_image_count == 0 {
            return self.file_bytes;
        }

        let scale = (options.scale as f64).clamp(0.10, 1.00);
        let scale_sq = scale * scale;
        let quality = (options.quality.clamp(1, 100)) as f64;
        // Keep filter multiplier smooth in (1.0 - scale) so est(scale) remains strictly monotone.
        let filter_mult = if options.resize_filter == ResizeFilterChoice::Lanczos3 {
            1.00 + 0.03 * (1.0 - scale)
        } else {
            1.00
        };

        let est_image_bytes = match options.format {
            ImageFormatChoice::Jpeg => {
                let bpp = self.sample_rgb_jpeg_q50_bpp * jpeg_quality_multiplier(quality) * filter_mult;
                let non_dct_count = self
                    .compressible_image_count
                    .saturating_sub(self.orig_dct_image_count) as f64;
                let non_dct_rgb = self
                    .compressible_rgb_pixels
                    .saturating_sub(self.orig_dct_rgb_pixels) as f64;
                let non_dct_gray = self
                    .compressible_gray_pixels
                    .saturating_sub(self.orig_dct_gray_pixels) as f64;
                let non_dct_eff = non_dct_rgb + GRAY_TO_RGB_JPEG_RATIO * non_dct_gray;
                let non_dct_est = (non_dct_eff * scale_sq * bpp) / 8.0 + non_dct_count * 620.0;

                let dct_eff = (self.orig_dct_rgb_pixels as f64)
                    + GRAY_TO_RGB_JPEG_RATIO * (self.orig_dct_gray_pixels as f64);
                let mut dct_est = (dct_eff * scale_sq * bpp) / 8.0
                    + (self.orig_dct_image_count as f64) * 620.0;
                if apply_orig_dct_cap && (scale - 1.0).abs() < 1e-4 && self.orig_dct_image_count > 0 {
                    dct_est = dct_est.min(self.orig_dct_bytes as f64);
                }

                non_dct_est + dct_est
            }
            ImageFormatChoice::Avif => {
                // AVIF pipeline encodes via ravif for stats and writes RGB /DCTDecode into the PDF.
                let bpp = self.sample_rgb_jpeg_q50_bpp * jpeg_quality_multiplier(quality) * filter_mult;
                let eff_pixels = (self.compressible_rgb_pixels + self.compressible_gray_pixels) as f64;
                (eff_pixels * scale_sq * bpp) / 8.0 + (self.compressible_image_count as f64) * 620.0
            }
            ImageFormatChoice::Png => {
                let bpp = self.sample_rgb_flate_bpp * filter_mult;
                let eff_pixels = (self.compressible_rgb_pixels as f64)
                    + GRAY_TO_RGB_FLATE_RATIO * (self.compressible_gray_pixels as f64);
                (eff_pixels * scale_sq * bpp) / 8.0 + (self.compressible_image_count as f64) * 64.0
            }
            ImageFormatChoice::Webp => {
                // WebP pipeline converts to RGB and writes lossless PDF /FlateDecode streams.
                let bpp = self.sample_rgb_flate_bpp * filter_mult;
                let eff_pixels = (self.compressible_rgb_pixels + self.compressible_gray_pixels) as f64;
                (eff_pixels * scale_sq * bpp) / 8.0 + (self.compressible_image_count as f64) * 64.0
            }
        };

        (self.non_image_overhead_bytes as f64 + est_image_bytes)
            .round()
            .max(0.0) as usize
    }

    /// Solves for `scale` (`0.10..=1.00` in `0.01` steps) whose estimated size best matches
    /// `target_bytes` while keeping `options.quality` and format fixed.
    pub fn solve_scale_for_target(&self, options: &CompressionOptions, target_bytes: usize) -> f32 {
        if self.compressible_image_count == 0 || target_bytes == 0 {
            return options.scale.clamp(0.10, 1.00);
        }

        let current_scale = (options.scale.clamp(0.10, 1.00) * 100.0).round() / 100.0;
        let mut probe = options.clone();
        probe.scale = current_scale;
        let mut best_scale = current_scale;
        let mut best_diff = self
            .estimate_compressed_bytes_impl(&probe, false)
            .abs_diff(target_bytes);

        // Evaluate all 0.01 scale steps; prefer higher scale on exact ties.
        for pct in (10..=100).rev() {
            let candidate = (pct as f32) / 100.0;
            probe.scale = candidate;
            let diff = self
                .estimate_compressed_bytes_impl(&probe, false)
                .abs_diff(target_bytes);
            if diff < best_diff {
                best_diff = diff;
                best_scale = candidate;
            }
        }

        best_scale
    }

    /// Solves for `quality` (`1..=100`) whose estimated size best matches `target_bytes`
    /// while keeping `options.scale` and format fixed.
    pub fn solve_quality_for_target(&self, options: &CompressionOptions, target_bytes: usize) -> u8 {
        if self.compressible_image_count == 0
            || target_bytes == 0
            || !options.format.uses_quality()
        {
            return options.quality.clamp(1, 100);
        }

        let current_q = options.quality.clamp(1, 100);
        let mut probe = options.clone();
        probe.quality = current_q;
        let mut best_q = current_q;
        let mut best_diff = self.estimate_compressed_bytes(&probe).abs_diff(target_bytes);

        for q in 1..=100 {
            probe.quality = q;
            let diff = self.estimate_compressed_bytes(&probe).abs_diff(target_bytes);
            if diff < best_diff {
                best_diff = diff;
                best_q = q;
            }
        }

        best_q
    }

    /// Adjusts `options.scale` first (`0.10..=1.00`), and if `scale` hits its lower or upper bound
    /// on a quality-configurable format, adjusts `options.quality` (`1..=100`) to approach `target_bytes`.
    pub fn apply_target_scale_first(&self, options: &mut CompressionOptions, target_bytes: usize) {
        if self.compressible_image_count == 0 || target_bytes == 0 {
            return;
        }

        let solved_scale = self.solve_scale_for_target(options, target_bytes);
        options.scale = solved_scale;

        if options.format.uses_quality() {
            let est_bytes = self.estimate_compressed_bytes(options);
            let hit_lower_and_over = solved_scale <= 0.105 && est_bytes > target_bytes;
            let hit_upper_and_under = solved_scale >= 0.995 && est_bytes < target_bytes;
            if hit_lower_and_over || hit_upper_and_under {
                options.quality = self.solve_quality_for_target(options, target_bytes);
            }
        }
    }
}

fn jpeg_quality_multiplier(q: f64) -> f64 {
    let q = q.clamp(1.0, 100.0);
    if q <= 50.0 {
        0.18 + 0.82 * (q / 50.0).powf(0.85)
    } else {
        let t = (q - 50.0) / 50.0;
        1.0 + 0.65 * t + 3.65 * t.powf(3.5)
    }
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub current: usize,
    pub total: usize,
    pub message: String,
}

#[derive(Debug, Clone, Default)]
pub struct CompressionStats {
    pub original_bytes: usize,
    pub compressed_bytes: usize,
    pub total_image_objects: usize,
    pub images_recompressed: usize,
    pub images_kept_original: usize,
    pub images_skipped_small_or_mask: usize,
    pub images_skipped_unsupported: usize,
    pub skipped_reasons: Vec<String>,
    pub intermediate_format_bytes: Option<usize>,
}

impl CompressionStats {
    pub fn saved_percentage(&self) -> f64 {
        if self.original_bytes == 0 {
            return 0.0;
        }
        let orig = self.original_bytes as f64;
        let comp = self.compressed_bytes as f64;
        ((orig - comp) / orig) * 100.0
    }

    pub fn exceeds_target_mb(&self, target_mb: f64) -> bool {
        let target_bytes = mb_to_bytes(target_mb);
        target_mb > 0.0 && target_bytes > 0 && self.compressed_bytes > target_bytes
    }
}

#[derive(Debug)]
struct ExtractedImage {
    image: DynamicImage,
    orig_w: u32,
    orig_h: u32,
    is_original_dct: bool,
    orig_raw_len: usize,
    had_smask: bool,
}

#[derive(Debug)]
struct EncodedPdfStream {
    bytes: Vec<u8>,
    filter_name: &'static [u8],
    color_space_name: &'static [u8],
    decode_parms: Option<Dictionary>,
    intermediate_bytes: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum BaseColorSpace {
    Gray,
    Rgb,
    Cmyk,
}

impl BaseColorSpace {
    fn components(self) -> usize {
        match self {
            Self::Gray => 1,
            Self::Rgb => 3,
            Self::Cmyk => 4,
        }
    }
}

enum ColorSpaceSpec {
    Direct(BaseColorSpace),
    Indexed {
        base: BaseColorSpace,
        hival: usize,
        lookup: Vec<u8>,
    },
}

enum SkipReason {
    Unsupported(String),
}

/// Inspects a PDF in memory without modifying it, recording layout/pixel metrics and
/// sample-calibrating at most one representative large image for instant size estimation.
pub fn inspect_pdf(input_bytes: &[u8]) -> Result<PdfSummary, PdfError> {
    let doc = load_unencrypted_document(input_bytes)?;
    let page_count = doc.get_pages().len();
    let subsidiary_masks = collect_subsidiary_mask_ids(&doc);

    let mut image_count = 0usize;
    let mut compressible_image_count = 0usize;
    let mut compressible_stream_bytes = 0usize;
    let mut compressible_smask_ids = HashSet::new();
    let mut non_compressible_smask_ids = HashSet::new();

    let mut compressible_rgb_pixels = 0u64;
    let mut compressible_gray_pixels = 0u64;
    let mut orig_dct_image_count = 0usize;
    let mut orig_dct_bytes = 0usize;
    let mut orig_dct_rgb_pixels = 0u64;
    let mut orig_dct_gray_pixels = 0u64;

    let mut best_sample: Option<(ObjectId, u32, u32, u64)> = None;

    for (&id, obj) in &doc.objects {
        if subsidiary_masks.contains(&id) {
            continue;
        }
        let Object::Stream(stream) = obj else {
            continue;
        };
        if !is_image_xobject(&doc, stream) {
            continue;
        }

        image_count += 1;

        let smask_ref_id = match stream.dict.get(b"SMask") {
            Ok(Object::Reference(ref_id)) => Some(*ref_id),
            _ => None,
        };

        let metadata = inspect_compressible_metadata(&doc, stream);
        let Some((w, h, is_gray, filters)) = metadata else {
            if let Some(ref_id) = smask_ref_id {
                non_compressible_smask_ids.insert(ref_id);
            }
            continue;
        };

        compressible_image_count += 1;
        compressible_stream_bytes = compressible_stream_bytes.saturating_add(stream.content.len());
        if let Some(ref_id) = smask_ref_id {
            compressible_smask_ids.insert(ref_id);
        }

        let pixels = (w as u64) * (h as u64);
        if is_gray {
            compressible_gray_pixels = compressible_gray_pixels.saturating_add(pixels);
        } else {
            compressible_rgb_pixels = compressible_rgb_pixels.saturating_add(pixels);
        }

        let is_orig_dct =
            filters.len() == 1 && filters[0] == b"DCTDecode" && stream.dict.get(b"SMask").is_err();
        if is_orig_dct {
            orig_dct_image_count += 1;
            orig_dct_bytes = orig_dct_bytes.saturating_add(stream.content.len());
            if is_gray {
                orig_dct_gray_pixels = orig_dct_gray_pixels.saturating_add(pixels);
            } else {
                orig_dct_rgb_pixels = orig_dct_rgb_pixels.saturating_add(pixels);
            }
        }

        if best_sample.is_none_or(|(_, _, _, best_px)| pixels > best_px) {
            best_sample = Some((id, w, h, pixels));
        }
    }

    for smask_id in compressible_smask_ids {
        if !non_compressible_smask_ids.contains(&smask_id) {
            if let Some(Object::Stream(smask_stream)) = doc.objects.get(&smask_id) {
                compressible_stream_bytes =
                    compressible_stream_bytes.saturating_add(smask_stream.content.len());
            }
        }
    }

    let non_image_overhead_bytes = input_bytes.len().saturating_sub(compressible_stream_bytes);

    let (mut sample_rgb_jpeg_q50_bpp, mut sample_rgb_flate_bpp) =
        (DEFAULT_RGB_JPEG_Q50_BPP, DEFAULT_RGB_FLATE_BPP);

    if let Some((sample_id, sw, sh, _)) = best_sample {
        if let Ok(extracted) = extract_image_from_doc(&doc, sample_id, sw, sh) {
            let (calibrated_jpeg, calibrated_flate) = calibrate_sample_image(&extracted.image);
            sample_rgb_jpeg_q50_bpp = calibrated_jpeg;
            sample_rgb_flate_bpp = calibrated_flate;
        }
    }

    Ok(PdfSummary {
        page_count,
        image_count,
        compressible_image_count,
        file_bytes: input_bytes.len(),
        non_image_overhead_bytes,
        compressible_rgb_pixels,
        compressible_gray_pixels,
        orig_dct_image_count,
        orig_dct_bytes,
        orig_dct_rgb_pixels,
        orig_dct_gray_pixels,
        sample_rgb_jpeg_q50_bpp,
        sample_rgb_flate_bpp,
    })
}

fn inspect_compressible_metadata(
    doc: &Document,
    stream: &Stream,
) -> Option<(u32, u32, bool, Vec<Vec<u8>>)> {
    if is_image_mask(doc, stream) {
        return None;
    }
    let (w, h) = read_image_dimensions(doc, stream)?;
    if w < MIN_IMAGE_DIMENSION || h < MIN_IMAGE_DIMENSION {
        return None;
    }
    if stream.dict.get(b"Mask").is_ok() {
        return None;
    }
    let filters = extract_filters(doc, &stream.dict).ok()?;
    validate_image_filters(&filters).ok()?;
    let cs = resolve_color_space_spec(doc, stream).ok()?;
    let is_gray = matches!(
        cs,
        ColorSpaceSpec::Direct(BaseColorSpace::Gray)
            | ColorSpaceSpec::Indexed {
                base: BaseColorSpace::Gray,
                ..
            }
    );
    Some((w, h, is_gray, filters))
}

fn calibrate_sample_image(img: &DynamicImage) -> (f64, f64) {
    let orig_pixels = (img.width() as u64) * (img.height() as u64);
    let owned_scaled;
    let sample = if orig_pixels > MAX_CALIBRATION_PIXELS {
        let ratio = (MAX_CALIBRATION_PIXELS as f64 / orig_pixels as f64).sqrt();
        let nw = ((img.width() as f64) * ratio).round().max(1.0) as u32;
        let nh = ((img.height() as f64) * ratio).round().max(1.0) as u32;
        owned_scaled = img.resize_exact(nw, nh, FilterType::Triangle);
        &owned_scaled
    } else {
        img
    };

    let w = sample.width();
    let h = sample.height();
    let px = (w as f64 * h as f64).max(1.0);
    let is_gray = matches!(sample, DynamicImage::ImageLuma8(_));

    let jpeg_bpp = match encode_jpeg_stream(sample, w, h, 50, None) {
        Ok(encoded) => {
            let payload = encoded.bytes.len().saturating_sub(600) as f64;
            let raw_bpp = (payload * 8.0 / px).max(0.03);
            if is_gray {
                raw_bpp / GRAY_TO_RGB_JPEG_RATIO
            } else {
                raw_bpp
            }
        }
        Err(_) => DEFAULT_RGB_JPEG_Q50_BPP,
    };

    let flate_bpp = {
        let (raw_bytes, ext_color): (Vec<u8>, ExtendedColorType) = match sample {
            DynamicImage::ImageLuma8(gray) => (gray.as_raw().clone(), ExtendedColorType::L8),
            _ => (sample.to_rgb8().into_raw(), ExtendedColorType::Rgb8),
        };
        let mut png_buf = Vec::new();
        let png_enc = PngEncoder::new_with_quality(
            &mut png_buf,
            CompressionType::Default,
            PngFilterType::Adaptive,
        );
        if png_enc.write_image(&raw_bytes, w, h, ext_color).is_ok() {
            let idat_len = extract_png_idat_stream(&png_buf)
                .map(|(_, data)| data.len())
                .unwrap_or_else(|| png_buf.len().saturating_sub(57));
            // Scale Default zlib (level 6) down ~4% to match CompressionType::Best used in export.
            let raw_bpp = (((idat_len as f64) * 0.96) * 8.0 / px).max(0.03);
            if is_gray {
                raw_bpp / GRAY_TO_RGB_FLATE_RATIO
            } else {
                raw_bpp
            }
        } else {
            DEFAULT_RGB_FLATE_BPP
        }
    };

    (jpeg_bpp, flate_bpp)
}

/// Compresses embedded images inside a PDF while leaving page boxes (`/MediaBox`, `/CropBox`)
/// and content streams (`cm` transformation matrices) untouched.
pub fn compress_pdf<F>(
    input_bytes: &[u8],
    options: &CompressionOptions,
    mut on_progress: F,
) -> Result<(Vec<u8>, CompressionStats), PdfError>
where
    F: FnMut(ProgressUpdate),
{
    let mut doc = load_unencrypted_document(input_bytes)?;
    let subsidiary_masks = collect_subsidiary_mask_ids(&doc);

    let mut candidate_ids = Vec::new();
    for (&id, obj) in &doc.objects {
        if subsidiary_masks.contains(&id) {
            continue;
        }
        if let Object::Stream(stream) = obj {
            if is_image_xobject(&doc, stream) {
                candidate_ids.push(id);
            }
        }
    }
    candidate_ids.sort_unstable();

    let total_candidates = candidate_ids.len();
    let mut stats = CompressionStats {
        original_bytes: input_bytes.len(),
        total_image_objects: total_candidates,
        ..Default::default()
    };

    let scale = options.scale.clamp(0.10, 1.00);
    let quality = options.quality.clamp(1, 100);
    let filter_type = options.resize_filter.to_image_filter();
    let mut total_intermediate: usize = 0;
    let mut has_intermediate = false;

    for (idx, id) in candidate_ids.iter().enumerate() {
        on_progress(ProgressUpdate {
            current: idx + 1,
            total: total_candidates.max(1),
            message: format!("Processing image {} of {}...", idx + 1, total_candidates),
        });

        // Read metadata and check skip rules using immutable &doc access first.
        let (is_mask, dims) = match doc.objects.get(id) {
            Some(Object::Stream(stream)) => {
                let is_mask = is_image_mask(&doc, stream);
                let dims = read_image_dimensions(&doc, stream);
                (is_mask, dims)
            }
            _ => continue,
        };

        if is_mask {
            stats.images_skipped_small_or_mask += 1;
            continue;
        }

        let (orig_w, orig_h) = match dims {
            Some((w, h)) if w >= MIN_IMAGE_DIMENSION && h >= MIN_IMAGE_DIMENSION => (w, h),
            Some(_) => {
                stats.images_skipped_small_or_mask += 1;
                continue;
            }
            None => {
                stats.images_skipped_unsupported += 1;
                push_unique_reason(
                    &mut stats.skipped_reasons,
                    "Skipped an image XObject with missing or invalid /Width or /Height."
                        .to_string(),
                );
                continue;
            }
        };

        let extracted = match extract_image_from_doc(&doc, *id, orig_w, orig_h) {
            Ok(img) => img,
            Err(SkipReason::Unsupported(reason)) => {
                stats.images_skipped_unsupported += 1;
                push_unique_reason(&mut stats.skipped_reasons, reason);
                continue;
            }
        };

        let (new_w, new_h, resized) = if (scale - 1.0).abs() < 1e-4 {
            (extracted.orig_w, extracted.orig_h, extracted.image)
        } else {
            let nw = ((extracted.orig_w as f32) * scale).round().max(1.0) as u32;
            let nh = ((extracted.orig_h as f32) * scale).round().max(1.0) as u32;
            let resized_img = extracted.image.resize_exact(nw, nh, filter_type);
            (nw, nh, resized_img)
        };

        let encoded = encode_for_pdf(&resized, new_w, new_h, options.format, quality)?;
        if let Some(inter) = encoded.intermediate_bytes {
            total_intermediate = total_intermediate.saturating_add(inter);
            has_intermediate = true;
        }

        // Never inflate a same-size, already-JPEG image when target is JPEG without SMask removal.
        if options.format == ImageFormatChoice::Jpeg
            && extracted.is_original_dct
            && new_w == extracted.orig_w
            && new_h == extracted.orig_h
            && !extracted.had_smask
            && encoded.bytes.len() >= extracted.orig_raw_len
        {
            stats.images_kept_original += 1;
            continue;
        }

        if let Some(Object::Stream(stream)) = doc.objects.get_mut(id) {
            apply_encoded_image_to_stream(stream, new_w, new_h, encoded);
            stats.images_recompressed += 1;
        }
    }

    if has_intermediate {
        stats.intermediate_format_bytes = Some(total_intermediate);
    }

    on_progress(ProgressUpdate {
        current: total_candidates.max(1),
        total: total_candidates.max(1),
        message: "Finalizing and optimizing PDF structure...".to_string(),
    });

    if !doc.is_encrypted() {
        doc.trailer.remove(b"Encrypt");
    }
    doc.prune_objects();
    doc.renumber_objects();
    doc.compress();

    let mut out_bytes = Vec::new();
    doc.save_modern(&mut out_bytes)
        .map_err(|e| PdfError::Save(e.to_string()))?;

    stats.compressed_bytes = out_bytes.len();
    Ok((out_bytes, stats))
}

fn load_unencrypted_document(input_bytes: &[u8]) -> Result<Document, PdfError> {
    let doc = Document::load_mem(input_bytes).map_err(|e| PdfError::Parse(e.to_string()))?;
    if doc.is_encrypted() {
        return Err(PdfError::Encrypted);
    }
    Ok(doc)
}

fn push_unique_reason(reasons: &mut Vec<String>, reason: String) {
    if reasons.len() < 8 && !reasons.iter().any(|r| r == &reason) {
        reasons.push(reason);
    }
}

fn resolve_obj<'a>(doc: &'a Document, obj: &'a Object) -> Option<&'a Object> {
    doc.dereference(obj).ok().map(|(_, resolved)| resolved)
}

fn dict_get_resolved<'a>(
    doc: &'a Document,
    dict: &'a Dictionary,
    key: &[u8],
) -> Option<&'a Object> {
    dict.get(key).ok().and_then(|o| resolve_obj(doc, o))
}

fn obj_as_i64(obj: &Object) -> Option<i64> {
    match obj {
        Object::Integer(v) => Some(*v),
        _ => None,
    }
}

fn obj_as_f64(obj: &Object) -> Option<f64> {
    match obj {
        Object::Real(v) => Some(*v as f64),
        Object::Integer(v) => Some(*v as f64),
        _ => None,
    }
}

fn obj_as_name(obj: &Object) -> Option<&[u8]> {
    match obj {
        Object::Name(name) => Some(name.as_slice()),
        _ => None,
    }
}

fn obj_as_bool(obj: &Object) -> Option<bool> {
    match obj {
        Object::Boolean(b) => Some(*b),
        _ => None,
    }
}

fn is_image_xobject(doc: &Document, stream: &Stream) -> bool {
    dict_get_resolved(doc, &stream.dict, b"Subtype")
        .and_then(obj_as_name)
        .is_some_and(|s| s == b"Image")
}

fn is_image_mask(doc: &Document, stream: &Stream) -> bool {
    dict_get_resolved(doc, &stream.dict, b"ImageMask")
        .and_then(obj_as_bool)
        .unwrap_or(false)
}

fn read_image_dimensions(doc: &Document, stream: &Stream) -> Option<(u32, u32)> {
    let w = dict_get_resolved(doc, &stream.dict, b"Width").and_then(obj_as_i64)?;
    let h = dict_get_resolved(doc, &stream.dict, b"Height").and_then(obj_as_i64)?;
    if w <= 0 || h <= 0 {
        return None;
    }
    let w_u32 = u32::try_from(w).ok()?;
    let h_u32 = u32::try_from(h).ok()?;
    Some((w_u32, h_u32))
}

/// Collects all object IDs that are referenced as `/SMask` or `/Mask` by any Image XObject
/// so we never re-encode a subsidiary grayscale soft-mask stream into RGB.
fn collect_subsidiary_mask_ids(doc: &Document) -> HashSet<ObjectId> {
    let mut mask_ids = HashSet::new();
    for obj in doc.objects.values() {
        if let Object::Stream(stream) = obj {
            if is_image_xobject(doc, stream) {
                for key in [b"SMask".as_slice(), b"Mask".as_slice()] {
                    if let Ok(Object::Reference(ref_id)) = stream.dict.get(key) {
                        mask_ids.insert(*ref_id);
                    }
                }
            }
        }
    }
    mask_ids
}

fn extract_filters(doc: &Document, dict: &Dictionary) -> Result<Vec<Vec<u8>>, SkipReason> {
    let Some(filter_obj) = dict_get_resolved(doc, dict, b"Filter") else {
        return Ok(Vec::new());
    };
    match filter_obj {
        Object::Name(name) => Ok(vec![name.clone()]),
        Object::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let resolved = resolve_obj(doc, item).ok_or_else(|| {
                    SkipReason::Unsupported("Unresolvable /Filter entry in image stream.".into())
                })?;
                let name = obj_as_name(resolved).ok_or_else(|| {
                    SkipReason::Unsupported("Invalid non-name /Filter entry in image stream.".into())
                })?;
                out.push(name.to_vec());
            }
            Ok(out)
        }
        _ => Err(SkipReason::Unsupported(
            "Invalid /Filter object type in image stream.".into(),
        )),
    }
}

fn resolve_object_shallow_for_stream(doc: &Document, obj: &Object) -> Object {
    match resolve_obj(doc, obj) {
        Some(Object::Array(arr)) => Object::Array(
            arr.iter()
                .map(|item| resolve_obj(doc, item).cloned().unwrap_or_else(|| item.clone()))
                .collect(),
        ),
        Some(other) => other.clone(),
        None => obj.clone(),
    }
}

fn decompress_stream_with_resolved_dict(
    doc: &Document,
    stream: &Stream,
    strip_last_filter: bool,
) -> Result<Vec<u8>, SkipReason> {
    let mut cloned = stream.clone();

    if let Ok(filter_obj) = stream.dict.get(b"Filter") {
        let resolved = resolve_object_shallow_for_stream(doc, filter_obj);
        if strip_last_filter {
            match resolved {
                Object::Array(mut arr) if arr.len() > 1 => {
                    arr.pop();
                    if arr.len() == 1 {
                        cloned.dict.set("Filter", arr.remove(0));
                    } else {
                        cloned.dict.set("Filter", Object::Array(arr));
                    }
                }
                _ => {
                    cloned.dict.remove(b"Filter");
                }
            }
        } else {
            cloned.dict.set("Filter", resolved);
        }
    }

    if let Ok(parms_obj) = stream.dict.get(b"DecodeParms") {
        let resolved = resolve_object_shallow_for_stream(doc, parms_obj);
        if strip_last_filter {
            match resolved {
                Object::Array(mut arr) if arr.len() > 1 => {
                    arr.pop();
                    if arr.len() == 1 {
                        cloned.dict.set("DecodeParms", arr.remove(0));
                    } else {
                        cloned.dict.set("DecodeParms", Object::Array(arr));
                    }
                }
                _ => {
                    cloned.dict.remove(b"DecodeParms");
                }
            }
        } else {
            cloned.dict.set("DecodeParms", resolved);
        }
    }

    if cloned.dict.get(b"Filter").is_err() {
        return Ok(cloned.content);
    }

    cloned
        .decompressed_content_with_limit(MAX_DECOMPRESSED_BYTES)
        .map_err(|e| SkipReason::Unsupported(format!("Stream decompression skipped: {e}")))
}

fn validate_image_filters(filters: &[Vec<u8>]) -> Result<bool, SkipReason> {
    for f in filters {
        if matches!(
            f.as_slice(),
            b"CCITTFaxDecode" | b"JBIG2Decode" | b"JPXDecode" | b"Crypt"
        ) {
            let name = String::from_utf8_lossy(f);
            return Err(SkipReason::Unsupported(format!(
                "Kept /{name} image unchanged (already specialized compression)."
            )));
        }
    }

    let last_is_dct = filters.last().is_some_and(|f| f == b"DCTDecode");
    let prefix_limit = if last_is_dct {
        filters.len().saturating_sub(1)
    } else {
        filters.len()
    };
    for f in &filters[..prefix_limit] {
        if !matches!(
            f.as_slice(),
            b"FlateDecode"
                | b"LZWDecode"
                | b"ASCII85Decode"
                | b"ASCIIHexDecode"
                | b"RunLengthDecode"
        ) {
            let name = String::from_utf8_lossy(f);
            return Err(SkipReason::Unsupported(format!(
                "Kept image with unsupported /{name} filter."
            )));
        }
    }

    Ok(last_is_dct)
}

fn extract_image_from_doc(
    doc: &Document,
    id: ObjectId,
    orig_w: u32,
    orig_h: u32,
) -> Result<ExtractedImage, SkipReason> {
    let stream = match doc.objects.get(&id) {
        Some(Object::Stream(s)) => s,
        _ => {
            return Err(SkipReason::Unsupported(
                "Image object is not a stream.".into(),
            ))
        }
    };

    // Preserve explicit chroma-key or stencil /Mask images untouched to avoid losing cutouts.
    if stream.dict.get(b"Mask").is_ok() {
        return Err(SkipReason::Unsupported(
            "Skipped an image using an explicit /Mask.".into(),
        ));
    }

    let filters = extract_filters(doc, &stream.dict)?;
    let last_is_dct = validate_image_filters(&filters)?;
    let orig_raw_len = stream.content.len();

    let mut base_img = if last_is_dct {
        let jpeg_bytes = if filters.len() == 1 {
            stream.content.clone()
        } else {
            decompress_stream_with_resolved_dict(doc, stream, true)?
        };
        decode_dct_image(doc, stream, &jpeg_bytes)?
    } else {
        let raw_samples = if filters.is_empty() {
            stream.content.clone()
        } else {
            decompress_stream_with_resolved_dict(doc, stream, false)?
        };
        decode_raw_pdf_samples(doc, stream, &raw_samples, orig_w, orig_h)?
    };

    // If the image has an /SMask soft mask, composite onto white before converting to opaque output.
    let mut had_smask = false;
    if let Some(smask_obj) = dict_get_resolved(doc, &stream.dict, b"SMask") {
        if let Object::Stream(smask_stream) = smask_obj {
            if let Some(alpha_img) = decode_smask_alpha(doc, smask_stream) {
                base_img = composite_over_white(&base_img, &alpha_img);
                had_smask = true;
            } else {
                return Err(SkipReason::Unsupported(
                    "Skipped an image with an unsupported /SMask stream.".into(),
                ));
            }
        }
    }

    Ok(ExtractedImage {
        image: base_img,
        orig_w,
        orig_h,
        is_original_dct: filters == vec![b"DCTDecode".to_vec()],
        orig_raw_len,
        had_smask,
    })
}

fn decode_dct_image(
    doc: &Document,
    stream: &Stream,
    jpeg_bytes: &[u8],
) -> Result<DynamicImage, SkipReason> {
    let mut img = image::load_from_memory_with_format(jpeg_bytes, ImageFormat::Jpeg).map_err(
        |e| SkipReason::Unsupported(format!("Skipped unreadable JPEG stream: {e}")),
    )?;

    // Check if an explicit inverted /Decode array exists on grayscale/RGB JPEGs.
    if let Some(decode_arr) = read_decode_array(doc, stream) {
        if matches!(img, DynamicImage::ImageLuma8(_))
            && decode_arr.len() >= 2
            && decode_arr[0] > decode_arr[1]
        {
            img.invert();
        }
    }

    Ok(img)
}

fn decode_smask_alpha(doc: &Document, smask_stream: &Stream) -> Option<GrayImage> {
    let (mw, mh) = read_image_dimensions(doc, smask_stream)?;
    let filters = extract_filters(doc, &smask_stream.dict).ok()?;
    let decoded = if filters == vec![b"DCTDecode".to_vec()] {
        image::load_from_memory_with_format(&smask_stream.content, ImageFormat::Jpeg)
            .ok()?
            .into_luma8()
    } else {
        for f in &filters {
            if !matches!(
                f.as_slice(),
                b"FlateDecode"
                    | b"LZWDecode"
                    | b"ASCII85Decode"
                    | b"ASCIIHexDecode"
                    | b"RunLengthDecode"
            ) {
                return None;
            }
        }
        let raw = if filters.is_empty() {
            smask_stream.content.clone()
        } else {
            decompress_stream_with_resolved_dict(doc, smask_stream, false).ok()?
        };
        let bpc = read_bits_per_component(doc, smask_stream);
        let decode_arr = read_decode_array(doc, smask_stream);
        decode_uncompressed_to_image(
            &raw,
            mw,
            mh,
            bpc,
            &ColorSpaceSpec::Direct(BaseColorSpace::Gray),
            decode_arr.as_deref(),
        )
        .ok()?
        .into_luma8()
    };
    Some(decoded)
}

fn composite_over_white(base: &DynamicImage, alpha_mask: &GrayImage) -> DynamicImage {
    let w = base.width();
    let h = base.height();
    let resized_alpha = if alpha_mask.width() != w || alpha_mask.height() != h {
        DynamicImage::ImageLuma8(alpha_mask.clone())
            .resize_exact(w, h, FilterType::Triangle)
            .into_luma8()
    } else {
        alpha_mask.clone()
    };

    let rgb = base.to_rgb8();
    let mut out = RgbImage::new(w, h);
    for (dst, (src, a)) in out
        .pixels_mut()
        .zip(rgb.pixels().zip(resized_alpha.pixels()))
    {
        let alpha = a.0[0] as u16;
        let inv = 255 - alpha;
        dst.0[0] = ((src.0[0] as u16 * alpha + 255 * inv + 127) / 255) as u8;
        dst.0[1] = ((src.0[1] as u16 * alpha + 255 * inv + 127) / 255) as u8;
        dst.0[2] = ((src.0[2] as u16 * alpha + 255 * inv + 127) / 255) as u8;
    }
    DynamicImage::ImageRgb8(out)
}

fn read_bits_per_component(doc: &Document, stream: &Stream) -> u8 {
    dict_get_resolved(doc, &stream.dict, b"BitsPerComponent")
        .and_then(obj_as_i64)
        .and_then(|v| u8::try_from(v).ok())
        .unwrap_or(8)
}

fn read_decode_array(doc: &Document, stream: &Stream) -> Option<Vec<f64>> {
    let arr = match dict_get_resolved(doc, &stream.dict, b"Decode")? {
        Object::Array(a) => a,
        _ => return None,
    };
    let mut values = Vec::with_capacity(arr.len());
    for item in arr {
        let resolved = resolve_obj(doc, item)?;
        values.push(obj_as_f64(resolved)?);
    }
    Some(values)
}

fn resolve_base_color_space(doc: &Document, obj: &Object) -> Result<BaseColorSpace, SkipReason> {
    let resolved = resolve_obj(doc, obj).ok_or_else(|| {
        SkipReason::Unsupported("Unresolvable /ColorSpace reference in image.".into())
    })?;

    match resolved {
        Object::Name(name) => match name.as_slice() {
            b"DeviceGray" | b"G" => Ok(BaseColorSpace::Gray),
            b"DeviceRGB" | b"RGB" => Ok(BaseColorSpace::Rgb),
            b"DeviceCMYK" | b"CMYK" => Ok(BaseColorSpace::Cmyk),
            other => Err(SkipReason::Unsupported(format!(
                "Kept image with /{} color space.",
                String::from_utf8_lossy(other)
            ))),
        },
        Object::Array(arr) if !arr.is_empty() => {
            let cs_type = resolve_obj(doc, &arr[0])
                .and_then(obj_as_name)
                .ok_or_else(|| {
                    SkipReason::Unsupported("Invalid array /ColorSpace in image.".into())
                })?;

            match cs_type {
                b"CalGray" => Ok(BaseColorSpace::Gray),
                b"CalRGB" => Ok(BaseColorSpace::Rgb),
                b"ICCBased" if arr.len() >= 2 => {
                    let profile_stream = match resolve_obj(doc, &arr[1]) {
                        Some(Object::Stream(s)) => s,
                        _ => {
                            return Err(SkipReason::Unsupported(
                                "Invalid /ICCBased profile stream.".into(),
                            ))
                        }
                    };
                    let n = dict_get_resolved(doc, &profile_stream.dict, b"N")
                        .and_then(obj_as_i64);
                    match n {
                        Some(1) => Ok(BaseColorSpace::Gray),
                        Some(3) => Ok(BaseColorSpace::Rgb),
                        Some(4) => Ok(BaseColorSpace::Cmyk),
                        _ => {
                            if let Ok(alt) = profile_stream.dict.get(b"Alternate") {
                                resolve_base_color_space(doc, alt)
                            } else {
                                Err(SkipReason::Unsupported(
                                    "Kept /ICCBased image with unsupported channel count.".into(),
                                ))
                            }
                        }
                    }
                }
                b"DeviceGray" | b"G" => Ok(BaseColorSpace::Gray),
                b"DeviceRGB" | b"RGB" => Ok(BaseColorSpace::Rgb),
                b"DeviceCMYK" | b"CMYK" => Ok(BaseColorSpace::Cmyk),
                other => Err(SkipReason::Unsupported(format!(
                    "Kept image with /{} color space.",
                    String::from_utf8_lossy(other)
                ))),
            }
        }
        _ => Err(SkipReason::Unsupported(
            "Unsupported /ColorSpace object in image.".into(),
        )),
    }
}

fn resolve_color_space_spec(doc: &Document, stream: &Stream) -> Result<ColorSpaceSpec, SkipReason> {
    let cs_obj = dict_get_resolved(doc, &stream.dict, b"ColorSpace").ok_or_else(|| {
        SkipReason::Unsupported("Skipped image without /ColorSpace.".into())
    })?;

    if let Object::Array(arr) = cs_obj {
        if let Some(cs_name) = arr.first().and_then(|o| resolve_obj(doc, o)).and_then(obj_as_name)
        {
            if (cs_name == b"Indexed" || cs_name == b"I") && arr.len() >= 4 {
                let base = resolve_base_color_space(doc, &arr[1])?;
                let hival = resolve_obj(doc, &arr[2])
                    .and_then(obj_as_i64)
                    .ok_or_else(|| {
                        SkipReason::Unsupported("Invalid /Indexed hival in image.".into())
                    })?;
                if !(0..=255).contains(&hival) {
                    return Err(SkipReason::Unsupported(
                        "Out-of-range /Indexed hival in image.".into(),
                    ));
                }
                let lookup_obj = resolve_obj(doc, &arr[3]).ok_or_else(|| {
                    SkipReason::Unsupported("Unresolvable /Indexed lookup table.".into())
                })?;
                let lookup = match lookup_obj {
                    Object::String(bytes, _) => bytes.clone(),
                    Object::Stream(s) => {
                        if s.dict.get(b"Filter").is_ok() {
                            decompress_stream_with_resolved_dict(doc, s, false)?
                        } else {
                            s.content.clone()
                        }
                    }
                    _ => {
                        return Err(SkipReason::Unsupported(
                            "Invalid /Indexed lookup object.".into(),
                        ))
                    }
                };
                return Ok(ColorSpaceSpec::Indexed {
                    base,
                    hival: hival as usize,
                    lookup,
                });
            }
        }
    }

    Ok(ColorSpaceSpec::Direct(resolve_base_color_space(
        doc, cs_obj,
    )?))
}

fn decode_raw_pdf_samples(
    doc: &Document,
    stream: &Stream,
    raw: &[u8],
    width: u32,
    height: u32,
) -> Result<DynamicImage, SkipReason> {
    let bpc = read_bits_per_component(doc, stream);
    let cs = resolve_color_space_spec(doc, stream)?;
    let decode_arr = read_decode_array(doc, stream);
    decode_uncompressed_to_image(raw, width, height, bpc, &cs, decode_arr.as_deref())
}

fn decode_uncompressed_to_image(
    raw: &[u8],
    width: u32,
    height: u32,
    bpc: u8,
    cs: &ColorSpaceSpec,
    decode_arr: Option<&[f64]>,
) -> Result<DynamicImage, SkipReason> {
    if !matches!(bpc, 1 | 2 | 4 | 8 | 16) {
        return Err(SkipReason::Unsupported(format!(
            "Kept image with unsupported {bpc}-bit depth."
        )));
    }

    match cs {
        ColorSpaceSpec::Indexed {
            base,
            hival,
            lookup,
        } => {
            if bpc == 16 {
                return Err(SkipReason::Unsupported(
                    "Kept /Indexed image with unsupported 16-bit indices.".into(),
                ));
            }
            let base_comps = base.components();
            let needed_table_len = (*hival + 1) * base_comps;
            if lookup.len() < needed_table_len {
                return Err(SkipReason::Unsupported(
                    "Skipped /Indexed image with truncated palette table.".into(),
                ));
            }
            let indices = unpack_samples_per_row(raw, width as usize, height as usize, 1, bpc)?;
            match base {
                BaseColorSpace::Gray => {
                    let mut out = Vec::with_capacity((width as usize) * (height as usize));
                    for idx_val in indices {
                        let idx = (idx_val as usize).min(*hival);
                        out.push(lookup[idx]);
                    }
                    let img = GrayImage::from_raw(width, height, out).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to build indexed grayscale buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageLuma8(img))
                }
                BaseColorSpace::Rgb => {
                    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 3);
                    for idx_val in indices {
                        let idx = (idx_val as usize).min(*hival) * 3;
                        out.extend_from_slice(&lookup[idx..idx + 3]);
                    }
                    let img = RgbImage::from_raw(width, height, out).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to build indexed RGB buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageRgb8(img))
                }
                BaseColorSpace::Cmyk => {
                    let mut out = Vec::with_capacity((width as usize) * (height as usize) * 3);
                    for idx_val in indices {
                        let idx = (idx_val as usize).min(*hival) * 4;
                        let (r, g, b) = cmyk_bytes_to_rgb(
                            lookup[idx],
                            lookup[idx + 1],
                            lookup[idx + 2],
                            lookup[idx + 3],
                        );
                        out.push(r);
                        out.push(g);
                        out.push(b);
                    }
                    let img = RgbImage::from_raw(width, height, out).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to build indexed CMYK buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageRgb8(img))
                }
            }
        }
        ColorSpaceSpec::Direct(base) => {
            let comps = base.components();
            let mut samples = unpack_samples_to_u8(raw, width as usize, height as usize, comps, bpc)?;
            if let Some(dec) = decode_arr {
                apply_decode_array_u8(&mut samples, comps, dec);
            }

            match base {
                BaseColorSpace::Gray => {
                    let img = GrayImage::from_raw(width, height, samples).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to assemble Grayscale pixel buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageLuma8(img))
                }
                BaseColorSpace::Rgb => {
                    let img = RgbImage::from_raw(width, height, samples).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to assemble RGB pixel buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageRgb8(img))
                }
                BaseColorSpace::Cmyk => {
                    let mut rgb_out = Vec::with_capacity((width as usize) * (height as usize) * 3);
                    for chunk in samples.chunks_exact(4) {
                        let (r, g, b) = cmyk_bytes_to_rgb(chunk[0], chunk[1], chunk[2], chunk[3]);
                        rgb_out.push(r);
                        rgb_out.push(g);
                        rgb_out.push(b);
                    }
                    let img = RgbImage::from_raw(width, height, rgb_out).ok_or_else(|| {
                        SkipReason::Unsupported("Failed to convert CMYK pixel buffer.".into())
                    })?;
                    Ok(DynamicImage::ImageRgb8(img))
                }
            }
        }
    }
}

fn cmyk_bytes_to_rgb(c: u8, m: u8, y: u8, k: u8) -> (u8, u8, u8) {
    let c = c as f32 / 255.0;
    let m = m as f32 / 255.0;
    let y = y as f32 / 255.0;
    let k = k as f32 / 255.0;
    let r = (255.0 * (1.0 - c) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
    let g = (255.0 * (1.0 - m) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
    let b = (255.0 * (1.0 - y) * (1.0 - k)).round().clamp(0.0, 255.0) as u8;
    (r, g, b)
}

fn apply_decode_array_u8(samples: &mut [u8], components: usize, decode: &[f64]) {
    if decode.len() < components * 2 {
        return;
    }
    // Check if decode is already the identity [0, 1, 0, 1, ...].
    let is_identity = (0..components).all(|c| {
        (decode[c * 2] - 0.0).abs() < 1e-4 && (decode[c * 2 + 1] - 1.0).abs() < 1e-4
    });
    if is_identity {
        return;
    }

    for pixel in samples.chunks_exact_mut(components) {
        for (c, byte) in pixel.iter_mut().enumerate() {
            let d_min = decode[c * 2];
            let d_max = decode[c * 2 + 1];
            let t = (*byte as f64) / 255.0;
            let mapped = d_min + t * (d_max - d_min);
            *byte = (mapped * 255.0).round().clamp(0.0, 255.0) as u8;
        }
    }
}

/// Unpacks raw PDF image samples (1, 2, 4, or 8 bpc) preserving exact integer values (used for /Indexed).
fn unpack_samples_per_row(
    raw: &[u8],
    width: usize,
    height: usize,
    components: usize,
    bpc: u8,
) -> Result<Vec<u8>, SkipReason> {
    let row_bits = width
        .checked_mul(components)
        .and_then(|v| v.checked_mul(bpc as usize))
        .ok_or_else(|| SkipReason::Unsupported("Image dimensions overflow.".into()))?;
    let row_bytes = row_bits.div_ceil(8);
    let expected_bytes = row_bytes
        .checked_mul(height)
        .ok_or_else(|| SkipReason::Unsupported("Image buffer size overflow.".into()))?;

    if raw.len() < expected_bytes {
        return Err(SkipReason::Unsupported(
            "Skipped image stream with truncated pixel data.".into(),
        ));
    }

    let total_samples = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(components))
        .ok_or_else(|| SkipReason::Unsupported("Image sample count overflow.".into()))?;

    if bpc == 8 {
        return Ok(raw[..total_samples].to_vec());
    }

    let mut out = Vec::with_capacity(total_samples);
    let mask = ((1u16 << bpc) - 1) as u8;
    let samples_per_row = width * components;

    for row in 0..height {
        let row_slice = &raw[row * row_bytes..(row + 1) * row_bytes];
        for i in 0..samples_per_row {
            let bit_offset = i * (bpc as usize);
            let byte_idx = bit_offset / 8;
            let shift = 8 - (bpc as usize) - (bit_offset % 8);
            let val = (row_slice[byte_idx] >> shift) & mask;
            out.push(val);
        }
    }
    Ok(out)
}

/// Unpacks raw PDF image samples (1, 2, 4, 8, or 16 bpc) and scales each sample into 0..=255.
fn unpack_samples_to_u8(
    raw: &[u8],
    width: usize,
    height: usize,
    components: usize,
    bpc: u8,
) -> Result<Vec<u8>, SkipReason> {
    let total_samples = width
        .checked_mul(height)
        .and_then(|v| v.checked_mul(components))
        .ok_or_else(|| SkipReason::Unsupported("Image sample count overflow.".into()))?;

    if bpc == 16 {
        let expected = total_samples
            .checked_mul(2)
            .ok_or_else(|| SkipReason::Unsupported("16-bit image buffer overflow.".into()))?;
        if raw.len() < expected {
            return Err(SkipReason::Unsupported(
                "Skipped 16-bit image stream with truncated pixel data.".into(),
            ));
        }
        let mut out = Vec::with_capacity(total_samples);
        for pair in raw[..expected].chunks_exact(2) {
            out.push(pair[0]);
        }
        return Ok(out);
    }

    let mut out = unpack_samples_per_row(raw, width, height, components, bpc)?;
    if bpc < 8 {
        let max_val = ((1u16 << bpc) - 1) as u16;
        for v in &mut out {
            *v = ((*v as u16 * 255 + max_val / 2) / max_val) as u8;
        }
    }
    Ok(out)
}

fn encode_for_pdf(
    image: &DynamicImage,
    width: u32,
    height: u32,
    format: ImageFormatChoice,
    quality: u8,
) -> Result<EncodedPdfStream, PdfError> {
    match format {
        ImageFormatChoice::Jpeg => encode_jpeg_stream(image, width, height, quality, None),
        ImageFormatChoice::Png => encode_pdf_flate_lossless(image, width, height, None),
        ImageFormatChoice::Webp => {
            // Pure-Rust lossless WebP pass via image-webp, then transcode to PDF-compatible /FlateDecode.
            let rgb = image.to_rgb8();
            let mut webp_buf = Vec::new();
            let webp_enc = WebPEncoder::new_lossless(&mut webp_buf);
            webp_enc
                .write_image(rgb.as_raw(), width, height, ExtendedColorType::Rgb8)
                .map_err(|e| PdfError::ImageEncode(format!("WebP encode failed: {e}")))?;

            let webp_len = webp_buf.len();
            let decoded = image::load_from_memory_with_format(&webp_buf, ImageFormat::WebP)
                .map_err(|e| PdfError::ImageEncode(format!("WebP decode failed: {e}")))?;
            encode_pdf_flate_lossless(&decoded, width, height, Some(webp_len))
        }
        ImageFormatChoice::Avif => {
            // Pure-Rust AVIF pass via ravif (no NASM/C toolchain required), then encode PDF-safe /DCTDecode.
            let rgb = image.to_rgb8();
            let mut avif_buf = Vec::new();
            let avif_enc = AvifEncoder::new_with_speed_quality(&mut avif_buf, 9, quality);
            avif_enc
                .write_image(rgb.as_raw(), width, height, ExtendedColorType::Rgb8)
                .map_err(|e| PdfError::ImageEncode(format!("AVIF encode failed: {e}")))?;

            let avif_len = avif_buf.len();
            encode_jpeg_stream(
                &DynamicImage::ImageRgb8(rgb),
                width,
                height,
                quality,
                Some(avif_len),
            )
        }
    }
}

fn encode_jpeg_stream(
    image: &DynamicImage,
    width: u32,
    height: u32,
    quality: u8,
    intermediate_bytes: Option<usize>,
) -> Result<EncodedPdfStream, PdfError> {
    let mut out = Vec::new();
    let mut encoder = JpegEncoder::new_with_quality(&mut out, quality);

    let color_space_name = match image {
        DynamicImage::ImageLuma8(gray) => {
            encoder
                .encode(gray.as_raw(), width, height, ExtendedColorType::L8)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG grayscale encode: {e}")))?;
            b"DeviceGray".as_slice()
        }
        _ => {
            let rgb = image.to_rgb8();
            encoder
                .encode(rgb.as_raw(), width, height, ExtendedColorType::Rgb8)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG RGB encode: {e}")))?;
            b"DeviceRGB".as_slice()
        }
    };

    Ok(EncodedPdfStream {
        bytes: out,
        filter_name: b"DCTDecode",
        color_space_name,
        decode_parms: None,
        intermediate_bytes,
    })
}

/// Encodes lossless pixels as a valid PDF `/FlateDecode` stream.
/// Uses PNG optimal row filtering (`PngEncoder` IDAT payload -> PDF `/Predictor 15`) when available,
/// falling back to raw zlib compression if PNG chunk inspection does not match 8-bit L/RGB.
fn encode_pdf_flate_lossless(
    image: &DynamicImage,
    width: u32,
    height: u32,
    intermediate_bytes: Option<usize>,
) -> Result<EncodedPdfStream, PdfError> {
    let (raw_bytes, ext_color, color_space_name): (Vec<u8>, ExtendedColorType, &'static [u8]) =
        match image {
            DynamicImage::ImageLuma8(gray) => (
                gray.as_raw().clone(),
                ExtendedColorType::L8,
                b"DeviceGray".as_slice(),
            ),
            _ => {
                let rgb = image.to_rgb8();
                (
                    rgb.into_raw(),
                    ExtendedColorType::Rgb8,
                    b"DeviceRGB".as_slice(),
                )
            }
        };

    let mut png_buf = Vec::new();
    let png_enc =
        PngEncoder::new_with_quality(&mut png_buf, CompressionType::Best, PngFilterType::Adaptive);
    png_enc
        .write_image(&raw_bytes, width, height, ext_color)
        .map_err(|e| PdfError::ImageEncode(format!("PNG encode: {e}")))?;

    if let Some((colors, idat_zlib)) = extract_png_idat_stream(&png_buf) {
        let decode_parms = dictionary! {
            "Predictor" => 15,
            "Colors" => colors as i64,
            "BitsPerComponent" => 8,
            "Columns" => width as i64,
        };
        return Ok(EncodedPdfStream {
            bytes: idat_zlib,
            filter_name: b"FlateDecode",
            color_space_name,
            decode_parms: Some(decode_parms),
            intermediate_bytes,
        });
    }

    let mut zlib = ZlibEncoder::new(Vec::new(), Compression::best());
    zlib.write_all(&raw_bytes)?;
    let compressed = zlib.finish()?;

    Ok(EncodedPdfStream {
        bytes: compressed,
        filter_name: b"FlateDecode",
        color_space_name,
        decode_parms: None,
        intermediate_bytes,
    })
}

/// Extracts the concatenated `IDAT` zlib stream from a valid non-interlaced 8-bit Gray/RGB PNG.
fn extract_png_idat_stream(png_bytes: &[u8]) -> Option<(u8, Vec<u8>)> {
    const SIG: &[u8] = b"\x89PNG\r\n\x1a\n";
    if !png_bytes.starts_with(SIG) {
        return None;
    }

    let mut pos = SIG.len();
    let mut colors: Option<u8> = None;
    let mut idat = Vec::new();

    while pos + 12 <= png_bytes.len() {
        let len = u32::from_be_bytes([
            png_bytes[pos],
            png_bytes[pos + 1],
            png_bytes[pos + 2],
            png_bytes[pos + 3],
        ]) as usize;
        let chunk_type = &png_bytes[pos + 4..pos + 8];
        let data_start = pos + 8;
        let data_end = data_start.checked_add(len)?;
        let crc_end = data_end.checked_add(4)?;
        if crc_end > png_bytes.len() {
            return None;
        }
        let data = &png_bytes[data_start..data_end];

        if chunk_type == b"IHDR" {
            if data.len() != 13 {
                return None;
            }
            let bit_depth = data[8];
            let color_type = data[9];
            let compression = data[10];
            let filter_method = data[11];
            let interlace = data[12];
            if bit_depth != 8 || compression != 0 || filter_method != 0 || interlace != 0 {
                return None;
            }
            colors = match color_type {
                0 => Some(1),
                2 => Some(3),
                _ => return None,
            };
        } else if chunk_type == b"IDAT" {
            idat.extend_from_slice(data);
        } else if chunk_type == b"IEND" {
            break;
        }

        pos = crc_end;
    }

    let colors = colors?;
    if idat.is_empty() {
        return None;
    }
    Some((colors, idat))
}

fn apply_encoded_image_to_stream(
    stream: &mut Stream,
    new_w: u32,
    new_h: u32,
    encoded: EncodedPdfStream,
) {
    stream.dict.set("Type", Object::Name(b"XObject".to_vec()));
    stream.dict.set("Subtype", Object::Name(b"Image".to_vec()));
    stream.dict.set("Width", Object::Integer(new_w as i64));
    stream.dict.set("Height", Object::Integer(new_h as i64));
    stream.dict.set("BitsPerComponent", Object::Integer(8));
    stream
        .dict
        .set("ColorSpace", Object::Name(encoded.color_space_name.to_vec()));
    stream
        .dict
        .set("Filter", Object::Name(encoded.filter_name.to_vec()));

    match encoded.decode_parms {
        Some(parms) => stream.dict.set("DecodeParms", Object::Dictionary(parms)),
        None => {
            stream.dict.remove(b"DecodeParms");
        }
    }

    // Remove stale stream metadata that no longer applies to the re-encoded RGB8/L8 payload.
    stream.dict.remove(b"SMask");
    stream.dict.remove(b"Mask");
    stream.dict.remove(b"Decode");
    stream.dict.remove(b"Matte");
    stream.dict.remove(b"SMaskInData");
    stream.dict.remove(b"Alternates");
    stream.dict.remove(b"OPI");

    stream
        .dict
        .set("Length", Object::Integer(encoded.bytes.len() as i64));
    stream.content = encoded.bytes;
    stream.allows_compression = false;
}

#[cfg(test)]
mod tests {
    use super::*;
    use lopdf::content::{Content, Operation};

    fn build_test_pdf_with_image(
        width: u32,
        height: u32,
        color_space: &str,
        raw_pixels: Vec<u8>,
        add_tiny_icon: bool,
    ) -> Vec<u8> {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();

        let mut zlib = ZlibEncoder::new(Vec::new(), Compression::default());
        zlib.write_all(&raw_pixels).expect("zlib write");
        let compressed_pixels = zlib.finish().expect("zlib finish");

        let img_stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => width as i64,
                "Height" => height as i64,
                "ColorSpace" => color_space,
                "BitsPerComponent" => 8,
                "Filter" => "FlateDecode",
                "Length" => compressed_pixels.len() as i64,
            },
            compressed_pixels,
        )
        .with_compression(false);
        let img_id = doc.add_object(img_stream);

        let mut xobjects = dictionary! {
            "Im1" => img_id,
        };

        if add_tiny_icon {
            let icon_bytes = vec![200u8; 20 * 20 * 3];
            let icon_stream = Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => 20,
                    "Height" => 20,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                    "Length" => icon_bytes.len() as i64,
                },
                icon_bytes,
            );
            let icon_id = doc.add_object(icon_stream);
            xobjects.set("ImTiny", icon_id);
        }

        let resources_id = doc.add_object(dictionary! {
            "XObject" => xobjects,
        });

        let content = Content {
            operations: vec![
                Operation::new("q", vec![]),
                Operation::new(
                    "cm",
                    vec![
                        612.into(),
                        0.into(),
                        0.into(),
                        792.into(),
                        0.into(),
                        0.into(),
                    ],
                ),
                Operation::new("Do", vec![Object::Name(b"Im1".to_vec())]),
                Operation::new("Q", vec![]),
            ],
        };
        let content_id = doc.add_object(Stream::new(
            dictionary! {},
            content.encode().expect("encode content"),
        ));

        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "CropBox" => vec![10.into(), 10.into(), 602.into(), 782.into()],
            "Resources" => resources_id,
            "Contents" => content_id,
        });

        let pages = dictionary! {
            "Type" => "Pages",
            "Kids" => vec![page_id.into()],
            "Count" => 1,
        };
        doc.objects.insert(pages_id, Object::Dictionary(pages));

        let catalog_id = doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc.trailer.set("Root", catalog_id);

        let mut buf = Vec::new();
        doc.save_to(&mut buf).expect("save test pdf");
        buf
    }

    #[test]
    fn preserves_layout_boxes_and_cm_while_downscaling_rgb_jpeg() {
        let width = 200u32;
        let height = 100u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                pixels.push(((x * 2) % 255) as u8);
                pixels.push(((y * 2) % 255) as u8);
                pixels.push(128);
            }
        }

        let input = build_test_pdf_with_image(width, height, "DeviceRGB", pixels, true);
        let opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.5,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };

        let (output, stats) = compress_pdf(&input, &opts, |_| {}).expect("compression succeeds");
        assert_eq!(stats.images_recompressed, 1);
        assert_eq!(stats.images_skipped_small_or_mask, 1);

        let out_doc = Document::load_mem(&output).expect("valid output pdf");
        let pages = out_doc.get_pages();
        assert_eq!(pages.len(), 1);
        let (_, &page_id) = pages.iter().next().expect("page exists");
        let page_dict = out_doc.get_dictionary(page_id).expect("page dict");

        // Verify MediaBox & CropBox preserved intact.
        let media_box = page_dict
            .get(b"MediaBox")
            .expect("MediaBox")
            .as_array()
            .expect("array");
        assert_eq!(
            media_box
                .iter()
                .map(|o| o.as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![0, 0, 612, 792]
        );
        let crop_box = page_dict
            .get(b"CropBox")
            .expect("CropBox")
            .as_array()
            .expect("array");
        assert_eq!(
            crop_box
                .iter()
                .map(|o| o.as_i64().unwrap())
                .collect::<Vec<_>>(),
            vec![10, 10, 602, 782]
        );

        // Verify cm matrix in content stream preserved intact.
        let content_bytes = out_doc.get_page_content(page_id);
        let decoded_content = Content::decode(&content_bytes).expect("decode content");
        let cm_op = decoded_content
            .operations
            .iter()
            .find(|op| op.operator == "cm")
            .expect("cm op present");
        assert_eq!(cm_op.operands.len(), 6);
        assert_eq!(cm_op.operands[0].as_i64().unwrap(), 612);
        assert_eq!(cm_op.operands[3].as_i64().unwrap(), 792);

        // Verify resized image XObject dimensions and tiny icon untouched.
        let mut found_resized = false;
        let mut found_tiny = false;
        for obj in out_doc.objects.values() {
            if let Object::Stream(stream) = obj {
                if is_image_xobject(&out_doc, stream) {
                    let (w, h) = read_image_dimensions(&out_doc, stream).unwrap();
                    if w == 100 && h == 50 {
                        found_resized = true;
                        assert_eq!(
                            stream.dict.get(b"Filter").unwrap().as_name().unwrap(),
                            b"DCTDecode"
                        );
                        assert_eq!(
                            stream.dict.get(b"ColorSpace").unwrap().as_name().unwrap(),
                            b"DeviceRGB"
                        );
                        assert!(stream.dict.get(b"DecodeParms").is_err());
                    } else if w == 20 && h == 20 {
                        found_tiny = true;
                    }
                }
            }
        }
        assert!(found_resized);
        assert!(found_tiny);
    }

    #[test]
    fn supports_all_output_formats_without_errors() {
        let pixels = vec![140u8; 100 * 100 * 3];
        let input = build_test_pdf_with_image(100, 100, "DeviceRGB", pixels, false);

        for fmt in ImageFormatChoice::ALL {
            let opts = CompressionOptions {
                format: fmt,
                quality: 50,
                scale: 0.5,
                resize_filter: ResizeFilterChoice::CatmullRom,
            };
            let (out, stats) = compress_pdf(&input, &opts, |_| {}).expect("compress succeeds");
            assert_eq!(stats.images_recompressed, 1);
            let parsed = Document::load_mem(&out).expect("output PDF must parse cleanly");
            assert_eq!(parsed.get_pages().len(), 1);
        }
    }

    #[test]
    fn converts_cmyk_and_preserves_grayscale_and_flattens_smask() {
        // Test CMYK -> DeviceRGB
        let cmyk_pixels = vec![0u8; 100 * 100 * 4];
        let cmyk_pdf = build_test_pdf_with_image(100, 100, "DeviceCMYK", cmyk_pixels, false);
        let opts = CompressionOptions::default();
        let (out_cmyk, stats_cmyk) =
            compress_pdf(&cmyk_pdf, &opts, |_| {}).expect("cmyk compression succeeds");
        assert_eq!(stats_cmyk.images_recompressed, 1);
        let parsed_cmyk = Document::load_mem(&out_cmyk).expect("valid output pdf");
        let has_device_rgb = parsed_cmyk.objects.values().any(|o| match o {
            Object::Stream(s) => {
                is_image_xobject(&parsed_cmyk, s)
                    && s.dict
                        .get(b"ColorSpace")
                        .and_then(|cs| cs.as_name())
                        .is_ok_and(|n| n == b"DeviceRGB")
            }
            _ => false,
        });
        assert!(has_device_rgb);

        // Test Grayscale -> DeviceGray
        let gray_pixels = vec![200u8; 100 * 100];
        let gray_pdf = build_test_pdf_with_image(100, 100, "DeviceGray", gray_pixels, false);
        let (out_gray, stats_gray) =
            compress_pdf(&gray_pdf, &opts, |_| {}).expect("gray compression succeeds");
        assert_eq!(stats_gray.images_recompressed, 1);
        let parsed_gray = Document::load_mem(&out_gray).expect("valid gray pdf");
        let has_device_gray = parsed_gray.objects.values().any(|o| match o {
            Object::Stream(s) => {
                is_image_xobject(&parsed_gray, s)
                    && s.dict
                        .get(b"ColorSpace")
                        .and_then(|cs| cs.as_name())
                        .is_ok_and(|n| n == b"DeviceGray")
            }
            _ => false,
        });
        assert!(has_device_gray);

        // Test SMask compositing removes SMask entry
        let mut doc = Document::load_mem(&build_test_pdf_with_image(
            100,
            100,
            "DeviceRGB",
            vec![0u8; 100 * 100 * 3],
            false,
        ))
        .expect("load base pdf");
        let smask_stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100,
                "Height" => 100,
                "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 8,
            },
            vec![128u8; 100 * 100],
        );
        let smask_id = doc.add_object(smask_stream);
        for obj in doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image")
                    && s.dict.get(b"ColorSpace").and_then(|v| v.as_name()).ok() == Some(b"DeviceRGB")
                {
                    s.dict.set("SMask", smask_id);
                }
            }
        }
        let mut with_smask = Vec::new();
        doc.save_to(&mut with_smask).expect("save smask pdf");
        let (out_smask, stats_smask) =
            compress_pdf(&with_smask, &opts, |_| {}).expect("smask compression succeeds");
        assert_eq!(stats_smask.images_recompressed, 1);
        let parsed_smask = Document::load_mem(&out_smask).expect("valid output");
        for obj in parsed_smask.objects.values() {
            if let Object::Stream(s) = obj {
                if is_image_xobject(&parsed_smask, s) {
                    assert!(s.dict.get(b"SMask").is_err());
                }
            }
        }
    }

    #[test]
    fn rejects_invalid_pdf_gracefully_without_panic() {
        let err = inspect_pdf(b"not a real pdf").expect_err("should fail cleanly");
        assert!(matches!(err, PdfError::Parse(_)));
    }

    #[test]
    fn target_size_helpers_handle_zero_and_thresholds_correctly() {
        let stats = CompressionStats {
            original_bytes: 5 * 1024 * 1024,
            compressed_bytes: 2 * 1024 * 1024 + 512,
            ..Default::default()
        };

        assert!(!stats.exceeds_target_mb(0.0));
        assert!(!stats.exceeds_target_mb(2.5));
        assert!(stats.exceeds_target_mb(2.0));

        let exact_2mb = CompressionStats {
            original_bytes: 5 * 1024 * 1024,
            compressed_bytes: 2 * 1024 * 1024,
            ..Default::default()
        };
        assert!(!exact_2mb.exceeds_target_mb(2.0));
    }

    #[test]
    fn size_estimator_scales_monotonically_and_tracks_real_output() {
        let width = 400u32;
        let height = 300u32;
        let mut pixels = Vec::with_capacity((width * height * 3) as usize);
        for y in 0..height {
            for x in 0..width {
                let v = ((x * 7 + y * 13) % 251) as u8;
                pixels.push(v);
                pixels.push(((x * 3) % 255) as u8);
                pixels.push(((y * 5) % 255) as u8);
            }
        }

        let input = build_test_pdf_with_image(width, height, "DeviceRGB", pixels, false);
        let summary = inspect_pdf(&input).expect("inspect succeeds");
        assert_eq!(summary.compressible_image_count, 1);
        assert_eq!(summary.compressible_rgb_pixels, (width * height) as u64);

        let est_q50_full = summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 1.0,
            resize_filter: ResizeFilterChoice::Lanczos3,
        });
        let est_q50_half = summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.5,
            resize_filter: ResizeFilterChoice::Lanczos3,
        });
        let est_q50_quarter = summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.25,
            resize_filter: ResizeFilterChoice::Lanczos3,
        });
        assert!(est_q50_full > est_q50_half);
        assert!(est_q50_half > est_q50_quarter);

        let est_q20 = summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 20,
            scale: 0.5,
            resize_filter: ResizeFilterChoice::Lanczos3,
        });
        let est_q90 = summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 90,
            scale: 0.5,
            resize_filter: ResizeFilterChoice::Lanczos3,
        });
        assert!(est_q90 > est_q50_half);
        assert!(est_q50_half > est_q20);

        let opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.5,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let (actual_out, _) = compress_pdf(&input, &opts, |_| {}).expect("compress succeeds");
        let actual_len = actual_out.len() as f64;
        let est_len = summary.estimate_compressed_bytes(&opts) as f64;
        assert!(
            est_len > actual_len * 0.4 && est_len < actual_len * 2.2,
            "estimate {est_len} vs actual {actual_len}"
        );

        // PDFs with no compressible images should estimate equal to input file size.
        let tiny_only = build_test_pdf_with_image(20, 20, "DeviceRGB", vec![128; 20 * 20 * 3], false);
        let tiny_summary = inspect_pdf(&tiny_only).expect("inspect tiny");
        assert_eq!(tiny_summary.compressible_image_count, 0);
        assert_eq!(tiny_summary.estimate_compressed_bytes(&opts), tiny_only.len());
    }

    #[test]
    fn inverse_coupling_solvers_adjust_scale_and_quality_cleanly() {
        let synthetic_summary = PdfSummary {
            page_count: 10,
            image_count: 10,
            compressible_image_count: 10,
            file_bytes: 10 * 1024 * 1024,
            non_image_overhead_bytes: 200 * 1024,
            compressible_rgb_pixels: 80_000_000,
            compressible_gray_pixels: 0,
            orig_dct_image_count: 0,
            orig_dct_bytes: 0,
            orig_dct_rgb_pixels: 0,
            orig_dct_gray_pixels: 0,
            sample_rgb_jpeg_q50_bpp: 1.0,
            sample_rgb_flate_bpp: 8.0,
        };
        let target_2mb = 2 * 1024 * 1024;

        // 1. Increasing quality should decrease solved scale to preserve ~2 MB target.
        let low_q_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 30,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let high_q_opts = CompressionOptions {
            quality: 85,
            ..low_q_opts.clone()
        };
        let scale_at_low_q = synthetic_summary.solve_scale_for_target(&low_q_opts, target_2mb);
        let scale_at_high_q = synthetic_summary.solve_scale_for_target(&high_q_opts, target_2mb);
        assert!(scale_at_low_q > scale_at_high_q);

        // 2. Increasing scale should decrease solved quality to preserve ~2 MB target.
        let small_scale_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.30,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let large_scale_opts = CompressionOptions {
            scale: 0.60,
            ..small_scale_opts.clone()
        };
        let q_at_small_scale =
            synthetic_summary.solve_quality_for_target(&small_scale_opts, target_2mb);
        let q_at_large_scale =
            synthetic_summary.solve_quality_for_target(&large_scale_opts, target_2mb);
        assert!(q_at_small_scale > q_at_large_scale);

        // 3. Target-first adjusts scale first, and only changes quality if scale hits bounds.
        let mut opts_mid = CompressionOptions::default();
        synthetic_summary.apply_target_scale_first(&mut opts_mid, target_2mb);
        assert_eq!(opts_mid.quality, 50);
        assert!(opts_mid.scale > 0.10 && opts_mid.scale < 1.00);

        // Target below 0.10x @ Q50 (~308 KB): hits 0.10x lower bound and lowers quality below 50.
        let mut opts_tiny_target = CompressionOptions::default();
        synthetic_summary.apply_target_scale_first(&mut opts_tiny_target, 250 * 1024);
        assert!((opts_tiny_target.scale - 0.10).abs() < 1e-4);
        assert!(opts_tiny_target.quality < 50);

        // Target above 1.00x @ Q50 (~10 MB): hits 1.00x upper bound and raises quality above 50.
        let mut opts_large_target = CompressionOptions::default();
        synthetic_summary.apply_target_scale_first(&mut opts_large_target, 15 * 1024 * 1024);
        assert!((opts_large_target.scale - 1.00).abs() < 1e-4);
        assert!(opts_large_target.quality > 50);

        // 4. Lossless PNG leaves quality untouched and only tunes scale.
        let mut opts_png = CompressionOptions {
            format: ImageFormatChoice::Png,
            quality: 42,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        synthetic_summary.apply_target_scale_first(&mut opts_png, target_2mb);
        assert_eq!(opts_png.quality, 42);
        assert!((opts_png.scale - 0.15).abs() < 0.05);
    }
}
