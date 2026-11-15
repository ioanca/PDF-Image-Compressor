use std::borrow::Cow;
use std::collections::{HashMap, HashSet};
use std::io::{Read, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use flate2::{read::ZlibDecoder, write::ZlibEncoder, Compression};
use hayro_ccitt::{DecodeSettings as CcittDecodeSettings, DecoderContext as CcittDecoderContext, EncodingMode};
use hayro_jbig2::{DecodeSettings as Jbig2DecodeSettings, File as Jbig2File, Globals as Jbig2Globals, Page as Jbig2Page};
use hayro_jpeg2000::{ColorSpace as JpxColorSpace, DecodeSettings as JpxDecodeSettings, DecoderContext as JpxDecoderContext, Image as JpxImage};
use image::codecs::png::{CompressionType, FilterType as PngFilterType, PngEncoder};
use image::{
    imageops::FilterType, DynamicImage, ExtendedColorType, GrayImage, ImageEncoder, RgbImage,
};
use lopdf::{dictionary, Dictionary, Document, Object, ObjectId, SaveOptions, Stream};
use rayon::prelude::*;
use sha2::{Digest, Sha256};
use thiserror::Error;

use crate::jpeg_enc::{decode_jpeg_to_image, encode_jpeg_gray, encode_jpeg_rgb};
use crate::jpx_enc::{encode_jpx_gray, encode_jpx_rgb};

/// Maximum uncompressed stream size (256 MiB) per image layer to guard against corrupt streams.
const MAX_DECOMPRESSED_BYTES: usize = 256 * 1024 * 1024;
/// Minimum pixel dimension; images strictly smaller in width or height are skipped (icons/artifacts).
const MIN_IMAGE_DIMENSION: u32 = 50;
/// Binary MiB divisor used consistently for MB readouts and target-size limits.
pub const BYTES_PER_MB: f64 = 1024.0 * 1024.0;
/// Maximum pixel budget for sample-image bitrate calibration (~1 MP keeps PDF loading fast).
const MAX_CALIBRATION_PIXELS: u64 = 1_000_000;
/// Grayscale-to-RGB bytes-per-pixel ratios for JPEG (luma-only) and FlateDecode streams.
const GRAY_TO_RGB_JPEG_RATIO: f64 = 0.72;
const GRAY_TO_RGB_FLATE_RATIO: f64 = 0.42;
/// Default calibrated RGB bitrates (bpp) when sample calibration cannot run.
const DEFAULT_RGB_JPEG_Q50_BPP: f64 = 1.05;
const DEFAULT_RGB_FLATE_BPP: f64 = 8.50;
/// Empirical high-ratio Flate re-compression factor (`flate2` / best deflate) for non-image
/// `/FlateDecode` streams (content streams, fonts, ToUnicode CMaps, ICC profiles).
const FLATE_RECOMPRESS_FACTOR: f64 = 0.88;
/// Empirical compression factor for PDF 1.5+ object streams (`ObjStm`) + binary `/XRef` tables.
const OBJSTM_XREF_COMPRESSION_FACTOR: f64 = 0.35;
/// Fixed uncompressed header, `/ObjStm`/`/XRef` stream wrappers, and `startxref`/`%%EOF`
/// framing emitted by `lopdf::Document::save_modern`.
const PDF_MODERN_CONTAINER_OVERHEAD_BYTES: usize = 240;

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
    Jpeg2000,
    Png,
}

impl ImageFormatChoice {
    pub const ALL: [Self; 3] = [Self::Jpeg, Self::Jpeg2000, Self::Png];

    pub fn label(self) -> &'static str {
        match self {
            Self::Jpeg => "JPG",
            Self::Jpeg2000 => "JPEG 2000",
            Self::Png => "PNG",
        }
    }

    pub fn uses_quality(self) -> bool {
        matches!(self, Self::Jpeg | Self::Jpeg2000)
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
            Self::Lanczos3 => "Fancy",
            Self::CatmullRom => "Fast",
        }
    }

    fn to_image_filter(self) -> FilterType {
        match self {
            Self::Lanczos3 => FilterType::Lanczos3,
            Self::CatmullRom => FilterType::CatmullRom,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
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
            quality: 100,
            scale: 1.00,
            resize_filter: ResizeFilterChoice::Lanczos3,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PdfSummary {
    pub page_count: usize,
    pub image_count: usize,
    pub compressible_image_count: usize,
    #[cfg(test)]
    pub skipped_1bit_count: usize,
    pub file_bytes: usize,
    pub non_image_overhead_bytes: usize,
    #[cfg(test)]
    pub fixed_image_bytes: usize,
    pub dedup_fixed_image_bytes: usize,
    pub dedup_flate_stream_bytes: usize,
    pub structural_dict_bytes: usize,
    pub other_non_image_stream_bytes: usize,
    #[cfg(test)]
    pub gc_removed_objects: usize,
    #[cfg(test)]
    pub dedup_merged_objects: usize,
    pub fixed_bytes: usize,
    pub compressible_rgb_pixels: u64,
    pub compressible_gray_pixels: u64,
    pub orig_dct_image_count: usize,
    pub orig_dct_bytes: usize,
    pub orig_dct_rgb_pixels: u64,
    pub orig_dct_gray_pixels: u64,
    pub sample_rgb_jpeg_q50_bpp: f64,
    pub sample_rgb_flate_bpp: f64,
}

impl Default for PdfSummary {
    fn default() -> Self {
        Self {
            page_count: 0,
            image_count: 0,
            compressible_image_count: 0,
            #[cfg(test)]
            skipped_1bit_count: 0,
            file_bytes: 0,
            non_image_overhead_bytes: 0,
            #[cfg(test)]
            fixed_image_bytes: 0,
            dedup_fixed_image_bytes: 0,
            dedup_flate_stream_bytes: 0,
            structural_dict_bytes: 0,
            other_non_image_stream_bytes: 0,
            #[cfg(test)]
            gc_removed_objects: 0,
            #[cfg(test)]
            dedup_merged_objects: 0,
            fixed_bytes: 0,
            compressible_rgb_pixels: 0,
            compressible_gray_pixels: 0,
            orig_dct_image_count: 0,
            orig_dct_bytes: 0,
            orig_dct_rgb_pixels: 0,
            orig_dct_gray_pixels: 0,
            sample_rgb_jpeg_q50_bpp: DEFAULT_RGB_JPEG_Q50_BPP,
            sample_rgb_flate_bpp: DEFAULT_RGB_FLATE_BPP,
        }
    }
}

impl PdfSummary {
    /// Computes predicted non-image + uncompressible fixed-stream byte floor after
    /// GC/dedup, Flate Level-9 recompression, and PDF 1.5+ ObjStm/XRef packing.
    fn structural_fixed_base_bytes(&self) -> f64 {
        let has_granular_breakdown = self.dedup_fixed_image_bytes > 0
            || self.dedup_flate_stream_bytes > 0
            || self.structural_dict_bytes > 0
            || self.other_non_image_stream_bytes > 0;
        if has_granular_breakdown {
            (self.dedup_fixed_image_bytes as f64)
                + (self.dedup_flate_stream_bytes as f64) * FLATE_RECOMPRESS_FACTOR
                + (self.structural_dict_bytes as f64) * OBJSTM_XREF_COMPRESSION_FACTOR
                + (self.other_non_image_stream_bytes as f64)
        } else if self.fixed_bytes > 0 || self.non_image_overhead_bytes == 0 {
            self.fixed_bytes as f64
        } else {
            self.non_image_overhead_bytes as f64
        }
    }

    /// Estimates the compressed PDF size in O(1) pure arithmetic using calibrated pixel bitrates
    /// and structural GC/dedup/Flate/ObjStm baselines.
    pub fn estimate_compressed_bytes(&self, options: &CompressionOptions) -> usize {
        self.estimate_compressed_bytes_impl(options, true)
    }

    fn estimate_compressed_bytes_impl(
        &self,
        options: &CompressionOptions,
        apply_orig_dct_cap: bool,
    ) -> usize {
        let fixed_base = self.structural_fixed_base_bytes();
        if self.compressible_image_count == 0 {
            let fixed_rounded = fixed_base.round().max(0.0) as usize;
            return if fixed_rounded > 0 {
                if self.file_bytes > 0 {
                    fixed_rounded.min(self.file_bytes)
                } else {
                    fixed_rounded
                }
            } else {
                self.file_bytes
            };
        }

        let scale = if options.scale.is_finite() {
            options.scale.clamp(0.10, 1.00)
        } else {
            1.00
        };
        let scale_sq = (scale as f64) * (scale as f64);
        let filter_mult = match options.resize_filter {
            ResizeFilterChoice::Lanczos3 => 1.04,
            ResizeFilterChoice::CatmullRom => 0.96,
        };

        let image_bytes = match options.format {
            ImageFormatChoice::Jpeg => {
                let q = options.quality.clamp(1, 100);
                let q_mult = jpeg_quality_multiplier(q);
                let bpp_rgb = (self.sample_rgb_jpeg_q50_bpp * q_mult * filter_mult).clamp(0.06, 10.5);

                let scaled_dct_rgb = (self.orig_dct_rgb_pixels as f64) * scale_sq;
                let scaled_dct_gray = (self.orig_dct_gray_pixels as f64) * scale_sq;
                let dct_eff_pixels = scaled_dct_rgb + GRAY_TO_RGB_JPEG_RATIO * scaled_dct_gray;
                let dct_raw_est = dct_eff_pixels * (bpp_rgb / 8.0)
                    + (self.orig_dct_image_count as f64) * 320.0;
                let dct_est = if apply_orig_dct_cap
                    && (scale - 1.0).abs() < 1e-4
                    && self.orig_dct_bytes > 0
                {
                    dct_raw_est.min(self.orig_dct_bytes as f64)
                } else {
                    dct_raw_est
                };

                let other_count = self
                    .compressible_image_count
                    .saturating_sub(self.orig_dct_image_count);
                let other_rgb = self
                    .compressible_rgb_pixels
                    .saturating_sub(self.orig_dct_rgb_pixels) as f64;
                let other_gray = self
                    .compressible_gray_pixels
                    .saturating_sub(self.orig_dct_gray_pixels) as f64;
                let other_eff = (other_rgb + GRAY_TO_RGB_JPEG_RATIO * other_gray) * scale_sq;
                let other_est = if other_count > 0 {
                    other_eff * (bpp_rgb / 8.0) + (other_count as f64) * 320.0
                } else {
                    0.0
                };

                dct_est + other_est
            }
            ImageFormatChoice::Jpeg2000 => {
                let q = options.quality.clamp(1, 100);
                let q_mult = jpeg_quality_multiplier(q);
                let bpp_rgb =
                    (self.sample_rgb_jpeg_q50_bpp * 0.78 * q_mult * filter_mult).clamp(0.045, 9.0);
                let eff_pixels = ((self.compressible_rgb_pixels as f64)
                    + GRAY_TO_RGB_JPEG_RATIO * (self.compressible_gray_pixels as f64))
                    * scale_sq;
                eff_pixels * (bpp_rgb / 8.0) + (self.compressible_image_count as f64) * 380.0
            }
            ImageFormatChoice::Png => {
                let bpp_rgb = (self.sample_rgb_flate_bpp * filter_mult).clamp(0.35, 24.0);
                let eff_pixels = ((self.compressible_rgb_pixels as f64)
                    + GRAY_TO_RGB_FLATE_RATIO * (self.compressible_gray_pixels as f64))
                    * scale_sq;
                eff_pixels * (bpp_rgb / 8.0) + (self.compressible_image_count as f64) * 96.0
            }
        };

        let est = (fixed_base + image_bytes).round().max(0.0) as usize;
        if apply_orig_dct_cap && self.file_bytes > 0 {
            est.min(self.file_bytes)
        } else {
            est
        }
    }

    /// Finds the scale (`0.10..=1.00` in `0.01` steps) that best meets `target_bytes`,
    /// preferring the highest scale that stays at or below `target_bytes`.
    pub fn solve_scale_for_target(&self, options: &CompressionOptions, target_bytes: usize) -> f32 {
        if self.compressible_image_count == 0 || target_bytes == 0 {
            return options.scale.clamp(0.10, 1.00);
        }

        let mut best_under: Option<i32> = None;
        let mut best_any_pct = 10i32;
        let mut best_any_diff = usize::MAX;
        let mut probe = options.clone();

        for pct in (10..=100).rev() {
            probe.scale = (pct as f32) / 100.0;
            let est = self.estimate_compressed_bytes_impl(&probe, false);
            if est <= target_bytes {
                best_under = Some(pct);
                break;
            }
            let diff = est.abs_diff(target_bytes);
            if diff < best_any_diff {
                best_any_diff = diff;
                best_any_pct = pct;
            }
        }

        let chosen_pct = best_under.unwrap_or(best_any_pct);
        (chosen_pct as f32) / 100.0
    }

    /// Finds the quality (`1..=100`) that best meets `target_bytes`,
    /// preferring the highest quality that stays at or below `target_bytes`.
    pub fn solve_quality_for_target(&self, options: &CompressionOptions, target_bytes: usize) -> u8 {
        if self.compressible_image_count == 0 || target_bytes == 0 || !options.format.uses_quality()
        {
            return options.quality.clamp(1, 100);
        }

        let mut best_under: Option<u8> = None;
        let mut best_any_q = 1u8;
        let mut best_any_diff = usize::MAX;
        let mut probe = options.clone();

        for q in (1..=100).rev() {
            probe.quality = q;
            let est = self.estimate_compressed_bytes(&probe);
            if est <= target_bytes {
                best_under = Some(q);
                break;
            }
            let diff = est.abs_diff(target_bytes);
            if diff < best_any_diff {
                best_any_diff = diff;
                best_any_q = q;
            }
        }

        best_under.unwrap_or(best_any_q)
    }

    /// Adjusts `options.scale` first (`0.10..=1.00`), and if `scale` hits `0.10x` (while still
    /// over budget) or `1.00x` (with unused budget) on a quality-configurable format, solves
    /// `options.quality` (`1..=100`).
    pub fn apply_target_scale_first(&self, options: &mut CompressionOptions, target_bytes: usize) {
        if self.compressible_image_count == 0 || target_bytes == 0 {
            return;
        }
        options.scale = self.solve_scale_for_target(options, target_bytes);
        let current_est = self.estimate_compressed_bytes(options);

        if options.format.uses_quality()
            && ((options.scale <= 0.105 && current_est > target_bytes)
                || (options.scale >= 0.995 && current_est < target_bytes))
        {
            options.quality = self.solve_quality_for_target(options, target_bytes);
        }
    }

    /// Builds a realistic synthetic `PdfSummary` for estimator & solver unit tests.
    #[cfg(test)]
    pub fn synthetic_for_test(
        page_count: usize,
        file_bytes: usize,
        fixed_bytes: usize,
        compressible_image_count: usize,
        sample_dim: u32,
        _sample_orig_bytes: usize,
        is_single_dct: bool,
    ) -> Self {
        let dim = sample_dim.max(64);
        let total_rgb_pixels = (dim as u64) * (dim as u64) * (compressible_image_count as u64);
        let orig_dct_bytes = if is_single_dct {
            file_bytes.saturating_sub(fixed_bytes)
        } else {
            0
        };
        Self {
            page_count,
            image_count: compressible_image_count,
            compressible_image_count,
            skipped_1bit_count: 0,
            file_bytes,
            non_image_overhead_bytes: fixed_bytes,
            fixed_image_bytes: 0,
            dedup_fixed_image_bytes: 0,
            dedup_flate_stream_bytes: 0,
            structural_dict_bytes: 0,
            other_non_image_stream_bytes: fixed_bytes,
            gc_removed_objects: 0,
            dedup_merged_objects: 0,
            fixed_bytes,
            compressible_rgb_pixels: total_rgb_pixels,
            compressible_gray_pixels: 0,
            orig_dct_image_count: if is_single_dct {
                compressible_image_count
            } else {
                0
            },
            orig_dct_bytes,
            orig_dct_rgb_pixels: if is_single_dct { total_rgb_pixels } else { 0 },
            orig_dct_gray_pixels: 0,
            sample_rgb_jpeg_q50_bpp: DEFAULT_RGB_JPEG_Q50_BPP,
            sample_rgb_flate_bpp: DEFAULT_RGB_FLATE_BPP,
        }
    }
}

fn jpeg_quality_multiplier(quality: u8) -> f64 {
    let q = (quality.clamp(1, 100)) as f64;
    if q <= 50.0 {
        0.16 + 0.84 * (q / 50.0).powf(1.25)
    } else {
        1.0 + 3.2 * ((q - 50.0) / 50.0).powf(2.1)
    }
}

#[derive(Debug, Clone)]
pub struct ProgressUpdate {
    pub current: usize,
    pub total: usize,
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
    pub objects_pruned_gc: usize,
    #[cfg(test)]
    pub objects_deduplicated: usize,
    pub flate_streams_recompressed: usize,
    pub skipped_reasons: Vec<String>,
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FinalImageCodec {
    RawSamples,
    Dct,
    Jpx,
    Jbig2,
    CcittFax,
}

#[derive(Debug)]
struct ExtractedImage {
    image: DynamicImage,
    orig_raw_len: usize,
    smask_id: Option<ObjectId>,
    stencil_mask_id: Option<ObjectId>,
}

#[derive(Debug)]
struct EncodedPdfStream {
    bytes: Vec<u8>,
    filter_name: &'static [u8],
    color_space_name: &'static [u8],
    decode_parms: Option<Dictionary>,
    bits_per_component: u8,
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

enum ColorSpaceSpec<'a> {
    Direct(BaseColorSpace),
    Indexed {
        base: BaseColorSpace,
        hival: usize,
        lookup: Cow<'a, [u8]>,
    },
}

#[derive(Debug)]
enum SkipReason {
    Unsupported(String),
}

/// Collects unique, deduplicated candidate Image XObject IDs reachable across pages, annotations,
/// Form XObjects, patterns, and transparency groups, falling back to a deduplicated object scan
/// for page-less PDFs. Subsidiary `/SMask` and `/Mask` streams are excluded so they are handled
/// in lockstep with their parent images.
fn collect_candidate_image_ids(doc: &Document) -> Vec<ObjectId> {
    let subsidiary_masks = collect_subsidiary_mask_ids(doc);
    let mut seen_image_ids = HashSet::new();
    let mut visited_containers = HashSet::new();
    let pages = doc.get_pages();

    for &page_id in pages.values() {
        if let Ok((res_opt, res_ids)) = doc.get_page_resources(page_id) {
            if let Some(res_dict) = res_opt {
                collect_images_from_resources(
                    doc,
                    res_dict,
                    &mut visited_containers,
                    &subsidiary_masks,
                    &mut seen_image_ids,
                );
            }
            for rid in res_ids {
                if visited_containers.insert(rid) {
                    if let Ok(Object::Dictionary(res_dict)) = doc.get_object(rid) {
                        collect_images_from_resources(
                            doc,
                            res_dict,
                            &mut visited_containers,
                            &subsidiary_masks,
                            &mut seen_image_ids,
                        );
                    }
                }
            }
        }
        if let Ok(page_dict) = doc.get_dictionary(page_id) {
            collect_images_from_page_annots(
                doc,
                page_dict,
                &mut visited_containers,
                &subsidiary_masks,
                &mut seen_image_ids,
            );
        }
    }

    if seen_image_ids.is_empty() {
        for (&id, obj) in &doc.objects {
            if subsidiary_masks.contains(&id) {
                continue;
            }
            if let Object::Stream(stream) = obj {
                if is_image_xobject(doc, stream) {
                    seen_image_ids.insert(id);
                }
            }
        }
    }

    let mut candidate_ids: Vec<ObjectId> = seen_image_ids.into_iter().collect();
    candidate_ids.sort_unstable();
    candidate_ids
}

fn collect_images_from_resources_obj(
    doc: &Document,
    res_obj: &Object,
    visited_containers: &mut HashSet<ObjectId>,
    subsidiary_masks: &HashSet<ObjectId>,
    seen_image_ids: &mut HashSet<ObjectId>,
) {
    match res_obj {
        Object::Reference(rid) => {
            if visited_containers.insert(*rid) {
                if let Ok(Object::Dictionary(d)) = doc.get_object(*rid) {
                    collect_images_from_resources(
                        doc,
                        d,
                        visited_containers,
                        subsidiary_masks,
                        seen_image_ids,
                    );
                }
            }
        }
        Object::Dictionary(d) => {
            collect_images_from_resources(
                doc,
                d,
                visited_containers,
                subsidiary_masks,
                seen_image_ids,
            );
        }
        _ => {}
    }
}

fn collect_images_from_resources(
    doc: &Document,
    res_dict: &Dictionary,
    visited_containers: &mut HashSet<ObjectId>,
    subsidiary_masks: &HashSet<ObjectId>,
    seen_image_ids: &mut HashSet<ObjectId>,
) {
    if let Some(Object::Dictionary(xobj_dict)) = dict_get_resolved(doc, res_dict, b"XObject") {
        for (_, val) in xobj_dict.iter() {
            let Ok(obj_id) = val.as_reference() else {
                continue;
            };
            let Ok(Object::Stream(stream)) = doc.get_object(obj_id) else {
                continue;
            };
            if is_image_xobject(doc, stream) {
                if !subsidiary_masks.contains(&obj_id) {
                    seen_image_ids.insert(obj_id);
                }
            } else if dict_get_resolved(doc, &stream.dict, b"Subtype").and_then(obj_as_name)
                == Some(b"Form")
                && visited_containers.insert(obj_id)
            {
                if let Ok(sub_res) = stream.dict.get(b"Resources") {
                    collect_images_from_resources_obj(
                        doc,
                        sub_res,
                        visited_containers,
                        subsidiary_masks,
                        seen_image_ids,
                    );
                }
            }
        }
    }

    if let Some(Object::Dictionary(pat_dict)) = dict_get_resolved(doc, res_dict, b"Pattern") {
        for (_, val) in pat_dict.iter() {
            if let Ok(pat_id) = val.as_reference() {
                if visited_containers.insert(pat_id) {
                    if let Ok(Object::Stream(pat_stream)) = doc.get_object(pat_id) {
                        if let Ok(sub_res) = pat_stream.dict.get(b"Resources") {
                            collect_images_from_resources_obj(
                                doc,
                                sub_res,
                                visited_containers,
                                subsidiary_masks,
                                seen_image_ids,
                            );
                        }
                    }
                }
            }
        }
    }

    if let Some(Object::Dictionary(gs_dict)) = dict_get_resolved(doc, res_dict, b"ExtGState") {
        for (_, val) in gs_dict.iter() {
            if let Some(Object::Dictionary(gs)) = resolve_obj(doc, val) {
                if let Some(Object::Dictionary(smask_dict)) = dict_get_resolved(doc, gs, b"SMask") {
                    if let Ok(Object::Reference(g_id)) = smask_dict.get(b"G") {
                        if visited_containers.insert(*g_id) {
                            if let Ok(Object::Stream(g_stream)) = doc.get_object(*g_id) {
                                if let Ok(sub_res) = g_stream.dict.get(b"Resources") {
                                    collect_images_from_resources_obj(
                                        doc,
                                        sub_res,
                                        visited_containers,
                                        subsidiary_masks,
                                        seen_image_ids,
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

fn collect_images_from_page_annots(
    doc: &Document,
    page_dict: &Dictionary,
    visited_containers: &mut HashSet<ObjectId>,
    subsidiary_masks: &HashSet<ObjectId>,
    seen_image_ids: &mut HashSet<ObjectId>,
) {
    let Some(Object::Array(annots)) = dict_get_resolved(doc, page_dict, b"Annots") else {
        return;
    };
    for annot_ref in annots {
        let Some(Object::Dictionary(annot)) = resolve_obj(doc, annot_ref) else {
            continue;
        };
        let Some(Object::Dictionary(ap)) = dict_get_resolved(doc, annot, b"AP") else {
            continue;
        };
        for key in [b"N".as_slice(), b"R".as_slice(), b"D".as_slice()] {
            let Ok(entry) = ap.get(key) else {
                continue;
            };
            collect_images_from_appearance_entry(
                doc,
                entry,
                visited_containers,
                subsidiary_masks,
                seen_image_ids,
            );
        }
    }
}

fn collect_images_from_appearance_entry(
    doc: &Document,
    obj: &Object,
    visited_containers: &mut HashSet<ObjectId>,
    subsidiary_masks: &HashSet<ObjectId>,
    seen_image_ids: &mut HashSet<ObjectId>,
) {
    match obj {
        Object::Reference(id) => {
            if !visited_containers.insert(*id) {
                return;
            }
            match doc.get_object(*id) {
                Ok(Object::Stream(s)) => {
                    if let Ok(sub_res) = s.dict.get(b"Resources") {
                        collect_images_from_resources_obj(
                            doc,
                            sub_res,
                            visited_containers,
                            subsidiary_masks,
                            seen_image_ids,
                        );
                    }
                }
                Ok(Object::Dictionary(d)) => {
                    for (_, v) in d.iter() {
                        collect_images_from_appearance_entry(
                            doc,
                            v,
                            visited_containers,
                            subsidiary_masks,
                            seen_image_ids,
                        );
                    }
                }
                _ => {}
            }
        }
        Object::Dictionary(d) => {
            for (_, v) in d.iter() {
                collect_images_from_appearance_entry(
                    doc,
                    v,
                    visited_containers,
                    subsidiary_masks,
                    seen_image_ids,
                );
            }
        }
        _ => {}
    }
}

#[derive(Debug, Default)]
struct StructureOptReport {
    gc_removed: usize,
    dedup_merged: usize,
    cached_flate_decoded: HashMap<ObjectId, Vec<u8>>,
}

fn zlib_decompress_with_limit(raw: &[u8], max_bytes: usize) -> Option<Vec<u8>> {
    let mut dec = ZlibDecoder::new(raw);
    let mut out = Vec::new();
    dec.by_ref()
        .take((max_bytes + 1) as u64)
        .read_to_end(&mut out)
        .ok()?;
    if out.len() <= max_bytes {
        Some(out)
    } else {
        None
    }
}

fn push_refs_from_object(obj: &Object, queue: &mut Vec<ObjectId>, seen: &mut HashSet<ObjectId>) {
    let mut stack: Vec<&Object> = vec![obj];
    while let Some(cur) = stack.pop() {
        match cur {
            Object::Reference(id) => {
                if seen.insert(*id) {
                    queue.push(*id);
                }
            }
            Object::Array(arr) => {
                for item in arr {
                    stack.push(item);
                }
            }
            Object::Dictionary(dict) => {
                for (_, v) in dict.iter() {
                    stack.push(v);
                }
            }
            Object::Stream(stream) => {
                for (_, v) in stream.dict.iter() {
                    stack.push(v);
                }
            }
            _ => {}
        }
    }
}

/// Walks the PDF object graph starting from the trailer's `/Root` catalog and `/Info` dictionary
/// (recursively traversing dictionaries, arrays, and streams), returning all reachable `ObjectId`s.
fn collect_reachable_object_ids(doc: &Document) -> HashSet<ObjectId> {
    let mut reachable = HashSet::new();
    let mut queue = Vec::new();

    if let Ok(root_obj) = doc.trailer.get(b"Root") {
        push_refs_from_object(root_obj, &mut queue, &mut reachable);
    }
    if let Ok(info_obj) = doc.trailer.get(b"Info") {
        push_refs_from_object(info_obj, &mut queue, &mut reachable);
    }
    if reachable.is_empty() {
        for (_, v) in doc.trailer.iter() {
            push_refs_from_object(v, &mut queue, &mut reachable);
        }
    }
    if reachable.is_empty() {
        return doc.objects.keys().copied().collect();
    }

    while let Some(id) = queue.pop() {
        if let Some(obj) = doc.objects.get(&id) {
            push_refs_from_object(obj, &mut queue, &mut reachable);
        }
    }

    reachable
}

/// Strips orphaned/unreachable indirect objects (such as prior incremental save generations).
fn garbage_collect_unreachable_objects(doc: &mut Document) -> usize {
    let reachable = collect_reachable_object_ids(doc);
    let before = doc.objects.len();
    doc.objects.retain(|id, _| reachable.contains(id));
    before.saturating_sub(doc.objects.len())
}

fn rewrite_refs_in_object(obj: &mut Object, redirect: &HashMap<ObjectId, ObjectId>) -> bool {
    let mut changed = false;
    let mut stack: Vec<&mut Object> = vec![obj];
    while let Some(cur) = stack.pop() {
        match cur {
            Object::Reference(id) => {
                if let Some(&target) = redirect.get(id) {
                    if *id != target {
                        *id = target;
                        changed = true;
                    }
                }
            }
            Object::Array(arr) => {
                for item in arr.iter_mut() {
                    stack.push(item);
                }
            }
            Object::Dictionary(dict) => {
                for (_, v) in dict.iter_mut() {
                    stack.push(v);
                }
            }
            Object::Stream(stream) => {
                for (_, v) in stream.dict.iter_mut() {
                    stack.push(v);
                }
            }
            _ => {}
        }
    }
    changed
}

fn rewrite_all_references(
    doc: &mut Document,
    redirect: &HashMap<ObjectId, ObjectId>,
) -> Vec<ObjectId> {
    for (_, v) in doc.trailer.iter_mut() {
        rewrite_refs_in_object(v, redirect);
    }
    let mut dirty_ids = Vec::new();
    for (&id, obj) in doc.objects.iter_mut() {
        if rewrite_refs_in_object(obj, redirect) {
            dirty_ids.push(id);
        }
    }
    for bookmark in doc.bookmark_table.values_mut() {
        if let Some(&target) = redirect.get(&bookmark.page) {
            bookmark.page = target;
        }
    }
    dirty_ids
}

fn hash_object_value_canonical(obj: &Object, hasher: &mut Sha256, depth: usize) {
    if depth > 64 {
        hasher.update(b"DMAX");
        return;
    }
    match obj {
        Object::Null => hasher.update(b"N"),
        Object::Boolean(b) => {
            hasher.update(b"B");
            hasher.update([*b as u8]);
        }
        Object::Integer(i) => {
            hasher.update(b"I");
            hasher.update(i.to_le_bytes());
        }
        Object::Real(f) => {
            hasher.update(b"F");
            let norm = if *f == 0.0 { 0.0f32 } else { *f };
            hasher.update(norm.to_bits().to_le_bytes());
        }
        Object::Name(n) => {
            hasher.update(b"M");
            hasher.update((n.len() as u64).to_le_bytes());
            hasher.update(n);
        }
        Object::String(s, _) => {
            hasher.update(b"T");
            hasher.update((s.len() as u64).to_le_bytes());
            hasher.update(s);
        }
        Object::Reference((num, gen)) => {
            hasher.update(b"R");
            hasher.update(num.to_le_bytes());
            hasher.update(gen.to_le_bytes());
        }
        Object::Array(arr) => {
            hasher.update(b"A");
            hasher.update((arr.len() as u64).to_le_bytes());
            for item in arr {
                hash_object_value_canonical(item, hasher, depth + 1);
            }
        }
        Object::Dictionary(dict) => {
            hash_dict_entries_canonical(dict, hasher, &[], depth + 1);
        }
        Object::Stream(stream) => {
            hasher.update(b"S");
            hash_dict_entries_canonical(&stream.dict, hasher, &[b"Length"], depth + 1);
            hasher.update((stream.content.len() as u64).to_le_bytes());
            hasher.update(&stream.content);
        }
    }
}

fn hash_dict_entries_canonical(
    dict: &Dictionary,
    hasher: &mut Sha256,
    skip_keys: &[&[u8]],
    depth: usize,
) {
    let mut entries: Vec<(&[u8], &Object)> = dict
        .iter()
        .filter_map(|(k, v)| {
            let ks = k.as_slice();
            if skip_keys.contains(&ks) {
                None
            } else {
                Some((ks, v))
            }
        })
        .collect();
    entries.sort_by(|a, b| a.0.cmp(b.0));

    hasher.update(b"D");
    hasher.update((entries.len() as u64).to_le_bytes());
    for (k, v) in entries {
        hasher.update((k.len() as u64).to_le_bytes());
        hasher.update(k);
        hash_object_value_canonical(v, hasher, depth);
    }
}

fn is_non_mergeable_object(doc: &Document, obj: &Object) -> bool {
    match obj {
        Object::Dictionary(dict) => matches!(
            dict_get_resolved(doc, dict, b"Type").and_then(obj_as_name),
            Some(b"Page" | b"Pages" | b"Catalog")
        ),
        Object::Stream(stream) => matches!(
            dict_get_resolved(doc, &stream.dict, b"Type").and_then(obj_as_name),
            Some(b"ObjStm" | b"XRef")
        ),
        _ => false,
    }
}

struct CachedStreamPayload {
    content_tag: &'static [u8],
    payload_len: u64,
    payload_digest: [u8; 32],
    normalize_flate: bool,
    prefer_rank: u8,
    raw_len: usize,
}

fn combine_stream_hash(dict: &Dictionary, payload: &CachedStreamPayload) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(b"STREAM");
    if payload.normalize_flate {
        hash_dict_entries_canonical(dict, &mut hasher, &[b"Length", b"Filter"], 1);
    } else {
        hash_dict_entries_canonical(dict, &mut hasher, &[b"Length"], 1);
    }
    hasher.update(payload.content_tag);
    hasher.update(payload.payload_len.to_le_bytes());
    hasher.update(payload.payload_digest);
    hasher.finalize().into()
}

fn hash_canonical_object_for_dedup(
    doc: &Document,
    obj: &Object,
    keep_flate_payload: bool,
) -> Option<([u8; 32], u8, usize, Option<CachedStreamPayload>, Option<Vec<u8>>)> {
    if is_non_mergeable_object(doc, obj) {
        return None;
    }

    match obj {
        Object::Stream(stream) => {
            let raw_len = stream.content.len();
            let filters = extract_filters(doc, &stream.dict).unwrap_or_default();
            let has_parms = stream.dict.has(b"DecodeParms");
            let is_single_flate = filters.len() == 1 && filters[0] == b"FlateDecode";
            let is_image_codec_stream = filters.last().is_some_and(|&f| {
                matches!(
                    f,
                    b"DCTDecode" | b"JPXDecode" | b"JBIG2Decode" | b"CCITTFaxDecode"
                )
            });

            let mut retained_flate: Option<Vec<u8>> = None;
            let (content_tag, logical_bytes): (&'static [u8], Cow<'_, [u8]>) = if filters.is_empty() {
                (b"D", Cow::Borrowed(&stream.content))
            } else if is_image_codec_stream {
                (b"R", Cow::Borrowed(&stream.content))
            } else if is_single_flate && !has_parms {
                if let Some(decoded) = zlib_decompress_with_limit(&stream.content, MAX_DECOMPRESSED_BYTES) {
                    (b"D", Cow::Owned(decoded))
                } else {
                    (b"R", Cow::Borrowed(&stream.content))
                }
            } else if let Ok(decoded) =
                stream.decompressed_content_with_limit(MAX_DECOMPRESSED_BYTES)
            {
                (b"D", Cow::Owned(decoded))
            } else {
                (b"R", Cow::Borrowed(&stream.content))
            };

            let normalize_flate = !has_parms
                && content_tag == b"D"
                && (filters.is_empty() || is_single_flate);
            let prefer_rank = if is_single_flate { 0 } else { 1 };

            let mut payload_hasher = Sha256::new();
            payload_hasher.update(&logical_bytes);
            let payload_digest: [u8; 32] = payload_hasher.finalize().into();
            let payload_len = logical_bytes.len() as u64;

            if keep_flate_payload
                && is_single_flate
                && !has_parms
                && content_tag == b"D"
                && stream.allows_compression
                && !stream.content.is_empty()
                && !is_metadata_stream(doc, stream)
            {
                if let Cow::Owned(decoded_vec) = logical_bytes {
                    retained_flate = Some(decoded_vec);
                }
            }

            let cached_payload = CachedStreamPayload {
                content_tag,
                payload_len,
                payload_digest,
                normalize_flate,
                prefer_rank,
                raw_len,
            };
            let digest = combine_stream_hash(&stream.dict, &cached_payload);
            Some((digest, prefer_rank, raw_len, Some(cached_payload), retained_flate))
        }
        other => {
            let mut hasher = Sha256::new();
            hasher.update(b"OBJ");
            hash_object_value_canonical(other, &mut hasher, 1);
            let digest: [u8; 32] = hasher.finalize().into();
            Some((digest, 1u8, 0usize, None, None))
        }
    }
}

/// Iteratively merges identical streams, dictionaries (fonts, ExtGState, ColorSpace, encodings),
/// and arrays across the document until fixed-point convergence.
fn deduplicate_objects_iterative(
    doc: &mut Document,
    keep_flate_payloads: bool,
) -> (usize, HashMap<ObjectId, Vec<u8>>) {
    let mut total_merged = 0usize;

    let initial: Vec<(
        ObjectId,
        [u8; 32],
        u8,
        usize,
        Option<CachedStreamPayload>,
        Option<Vec<u8>>,
    )> = {
        let doc_ref = &*doc;
        doc.objects
            .par_iter()
            .filter_map(|(&id, obj)| {
                hash_canonical_object_for_dedup(doc_ref, obj, keep_flate_payloads).map(
                    |(hash, pref, raw_len, payload, decoded)| {
                        (id, hash, pref, raw_len, payload, decoded)
                    },
                )
            })
            .collect()
    };

    let mut object_hashes: HashMap<ObjectId, ([u8; 32], u8, usize)> =
        HashMap::with_capacity(initial.len());
    let mut stream_payloads: HashMap<ObjectId, CachedStreamPayload> = HashMap::new();
    let mut flate_decoded: HashMap<ObjectId, Vec<u8>> = HashMap::new();

    for (id, hash, pref, raw_len, payload, decoded) in initial {
        object_hashes.insert(id, (hash, pref, raw_len));
        if let Some(p) = payload {
            stream_payloads.insert(id, p);
        }
        if let Some(d) = decoded {
            flate_decoded.insert(id, d);
        }
    }

    for _round in 0..16 {
        let mut entries: Vec<(ObjectId, [u8; 32], u8, usize)> = object_hashes
            .iter()
            .map(|(&id, &(hash, pref, raw_len))| (id, hash, pref, raw_len))
            .collect();

        // Deterministic master choice: prefer compressed stream / smaller raw payload / lowest ID.
        entries.sort_unstable_by(|a, b| {
            a.2.cmp(&b.2).then_with(|| a.3.cmp(&b.3)).then_with(|| a.0.cmp(&b.0))
        });

        let mut seen: HashMap<[u8; 32], ObjectId> = HashMap::with_capacity(entries.len());
        let mut redirect: HashMap<ObjectId, ObjectId> = HashMap::new();

        for (id, hash, _, _) in entries {
            if let Some(&master_id) = seen.get(&hash) {
                if master_id != id {
                    redirect.insert(id, master_id);
                }
            } else {
                seen.insert(hash, id);
            }
        }

        if redirect.is_empty() {
            break;
        }

        total_merged += redirect.len();
        let dirty_ids = rewrite_all_references(doc, &redirect);
        for dup_id in redirect.keys() {
            doc.objects.remove(dup_id);
            object_hashes.remove(dup_id);
            stream_payloads.remove(dup_id);
            flate_decoded.remove(dup_id);
        }

        if dirty_ids.is_empty() {
            break;
        }

        for dirty_id in dirty_ids {
            let Some(obj) = doc.objects.get(&dirty_id) else {
                continue;
            };
            if is_non_mergeable_object(doc, obj) {
                continue;
            }
            match obj {
                Object::Stream(stream) => {
                    if let Some(payload) = stream_payloads.get(&dirty_id) {
                        let hash = combine_stream_hash(&stream.dict, payload);
                        object_hashes
                            .insert(dirty_id, (hash, payload.prefer_rank, payload.raw_len));
                    }
                }
                other => {
                    let mut hasher = Sha256::new();
                    hasher.update(b"OBJ");
                    hash_object_value_canonical(other, &mut hasher, 1);
                    let hash: [u8; 32] = hasher.finalize().into();
                    object_hashes.insert(dirty_id, (hash, 1u8, 0usize));
                }
            }
        }
    }

    (total_merged, flate_decoded)
}

/// Runs Priority 1 structural cleanup: strips trailer stale keys, performs reachability garbage
/// collection from `/Root` and `/Info`, and deduplicates identical streams and dictionaries.
fn optimize_structure_gc_and_dedup(
    doc: &mut Document,
    keep_flate_payloads: bool,
) -> StructureOptReport {
    if !doc.is_encrypted() {
        doc.trailer.remove(b"Encrypt");
    }
    doc.trailer.remove(b"Prev");
    doc.trailer.remove(b"XRefStm");

    let mut report = StructureOptReport::default();
    report.gc_removed += garbage_collect_unreachable_objects(doc);
    let (merged, mut cached_flate_decoded) =
        deduplicate_objects_iterative(doc, keep_flate_payloads);
    report.dedup_merged += merged;
    report.gc_removed += garbage_collect_unreachable_objects(doc);
    if !cached_flate_decoded.is_empty() {
        cached_flate_decoded.retain(|id, _| doc.objects.contains_key(id));
    }
    report.cached_flate_decoded = cached_flate_decoded;
    report
}

fn is_metadata_stream(doc: &Document, stream: &Stream) -> bool {
    dict_get_resolved(doc, &stream.dict, b"Type").and_then(obj_as_name) == Some(b"Metadata")
}

/// Priority 3: Re-compresses `/FlateDecode` streams (preserving any `/DecodeParms`) and
/// compresses uncompressed non-metadata streams using highest-ratio zlib (`Compression::best()`),
/// guarded strictly so only smaller streams replace the originals.
fn recompress_flate_and_raw_streams(
    doc: &mut Document,
    mut cached_flate_decoded: HashMap<ObjectId, Vec<u8>>,
) -> usize {
    enum StreamTaskPayload<'a> {
        RecompressFlate {
            raw: &'a [u8],
            pre_decoded: Option<Vec<u8>>,
        },
        CompressRaw {
            raw: &'a [u8],
        },
    }

    let updates: Vec<(ObjectId, Vec<u8>, bool)> = {
        let candidates: Vec<(ObjectId, StreamTaskPayload<'_>)> = doc
            .objects
            .iter()
            .filter_map(|(&id, obj)| {
                let Object::Stream(stream) = obj else {
                    return None;
                };
                if !stream.allows_compression || is_metadata_stream(doc, stream) {
                    return None;
                }
                let type_name = dict_get_resolved(doc, &stream.dict, b"Type").and_then(obj_as_name);
                if matches!(type_name, Some(b"ObjStm" | b"XRef")) {
                    return None;
                }
                let filters = extract_filters(doc, &stream.dict).ok()?;
                if filters.len() == 1 && filters[0] == b"FlateDecode" && !stream.content.is_empty()
                {
                    let pre_decoded = if stream.dict.has(b"DecodeParms") {
                        None
                    } else {
                        cached_flate_decoded.remove(&id)
                    };
                    Some((
                        id,
                        StreamTaskPayload::RecompressFlate {
                            raw: stream.content.as_slice(),
                            pre_decoded,
                        },
                    ))
                } else if filters.is_empty() && stream.content.len() > 24 {
                    Some((
                        id,
                        StreamTaskPayload::CompressRaw {
                            raw: stream.content.as_slice(),
                        },
                    ))
                } else {
                    None
                }
            })
            .collect();

        candidates
            .par_iter()
            .filter_map(|(id, task)| match task {
                StreamTaskPayload::RecompressFlate { raw, pre_decoded } => {
                    let owned_decoded;
                    let decoded_slice: &[u8] = match pre_decoded {
                        Some(buf) => buf.as_slice(),
                        None => {
                            // Decode raw zlib stream directly without stripping /DecodeParms predictors
                            // so re-compression remains bit-exact with existing /DecodeParms dictionaries.
                            owned_decoded = zlib_decompress_with_limit(raw, MAX_DECOMPRESSED_BYTES)?;
                            owned_decoded.as_slice()
                        }
                    };
                    let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
                    enc.write_all(decoded_slice).ok()?;
                    let recompressed = enc.finish().ok()?;
                    if recompressed.len() < raw.len() {
                        Some((*id, recompressed, false))
                    } else {
                        None
                    }
                }
                StreamTaskPayload::CompressRaw { raw } => {
                    let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
                    enc.write_all(raw).ok()?;
                    let compressed = enc.finish().ok()?;
                    if compressed.len() + 19 < raw.len() {
                        Some((*id, compressed, true))
                    } else {
                        None
                    }
                }
            })
            .collect()
    };

    let count = updates.len();
    for (id, compressed, add_filter) in updates {
        if let Some(Object::Stream(stream)) = doc.objects.get_mut(&id) {
            stream.set_content(compressed);
            if add_filter {
                stream.dict.set("Filter", "FlateDecode");
            }
        }
    }
    count
}

fn ensure_min_pdf_version(doc: &mut Document, min_major: u32, min_minor: u32) {
    let parsed = doc.version.split_once('.').and_then(|(maj, min)| {
        Some((
            maj.trim().parse::<u32>().ok()?,
            min.trim().parse::<u32>().ok()?,
        ))
    });
    if parsed.is_none_or(|ver| ver < (min_major, min_minor)) {
        doc.version = format!("{min_major}.{min_minor}");
    }
}

fn normalize_generations_to_zero(doc: &mut Document) {
    let non_zero_ids: Vec<ObjectId> = doc
        .objects
        .keys()
        .copied()
        .filter(|&(_, gen)| gen != 0)
        .collect();
    if non_zero_ids.is_empty() {
        return;
    }

    let mut next_num = doc.max_id.max(
        doc.objects
            .keys()
            .map(|&(num, _)| num)
            .max()
            .unwrap_or(0),
    );
    let mut redirect = HashMap::with_capacity(non_zero_ids.len());
    for old_id @ (num, _) in non_zero_ids {
        let candidate = (num, 0u16);
        let new_id = if !doc.objects.contains_key(&candidate)
            && !redirect.values().any(|&v| v == candidate)
        {
            candidate
        } else {
            loop {
                next_num = next_num.saturating_add(1);
                let fresh = (next_num, 0u16);
                if !doc.objects.contains_key(&fresh) {
                    break fresh;
                }
            }
        };
        redirect.insert(old_id, new_id);
    }

    doc.max_id = next_num;
    rewrite_all_references(doc, &redirect);
    let moved: Vec<(ObjectId, Object)> = redirect
        .iter()
        .filter_map(|(old_id, new_id)| doc.objects.remove(old_id).map(|obj| (*new_id, obj)))
        .collect();
    for (new_id, obj) in moved {
        doc.objects.insert(new_id, obj);
    }
}

fn find_subslice(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(0);
    }
    haystack.windows(needle.len()).position(|w| w == needle)
}

fn find_subslice_rev(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    if needle.is_empty() {
        return Some(haystack.len());
    }
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

/// Applies `FlateDecode` (`Compression::best()`) to the trailing `/XRef` cross-reference stream
/// written by `lopdf` when smaller. Because every object offset precedes `xref_start` and the
/// `/XRef` object header remains anchored at `xref_start`, all cross-reference offsets stay exact.
fn compress_trailing_xref_stream(pdf_bytes: &mut Vec<u8>) {
    let tail_start = pdf_bytes.len().saturating_sub(256);
    let Some(startxref_rel) = find_subslice_rev(&pdf_bytes[tail_start..], b"startxref") else {
        return;
    };
    let startxref_pos = tail_start + startxref_rel;
    let after_kw = &pdf_bytes[startxref_pos + b"startxref".len()..];
    let Some(eof_rel) = find_subslice(after_kw, b"%%EOF") else {
        return;
    };
    let offset_str = String::from_utf8_lossy(&after_kw[..eof_rel]);
    let Ok(xref_start) = offset_str.trim().parse::<usize>() else {
        return;
    };
    if xref_start >= startxref_pos {
        return;
    }

    let obj_slice = &pdf_bytes[xref_start..startxref_pos];
    let Some(stream_kw_rel) = find_subslice(obj_slice, b">>stream\n") else {
        return;
    };
    let dict_end_rel = stream_kw_rel;
    let Some(dict_start_rel) = find_subslice(&obj_slice[..dict_end_rel], b"<<") else {
        return;
    };
    let dict_bytes = &obj_slice[dict_start_rel..dict_end_rel + 2];
    let is_xref = find_subslice(dict_bytes, b"/Type/XRef").is_some()
        || find_subslice(dict_bytes, b"/Type /XRef").is_some();
    if !is_xref || find_subslice(dict_bytes, b"/Filter").is_some() {
        return;
    }

    let Some(len_kw_rel) = find_subslice(dict_bytes, b"/Length") else {
        return;
    };
    let mut cursor = len_kw_rel + b"/Length".len();
    while cursor < dict_bytes.len() && dict_bytes[cursor].is_ascii_whitespace() {
        cursor += 1;
    }
    let digits_start = cursor;
    while cursor < dict_bytes.len() && dict_bytes[cursor].is_ascii_digit() {
        cursor += 1;
    }
    if cursor == digits_start {
        return;
    }
    let digits_end = cursor;
    let Ok(raw_len) = std::str::from_utf8(&dict_bytes[digits_start..digits_end])
        .unwrap_or("")
        .parse::<usize>()
    else {
        return;
    };

    let data_start = stream_kw_rel + b">>stream\n".len();
    let Some(data_end) = data_start.checked_add(raw_len) else {
        return;
    };
    if data_end > obj_slice.len() || !obj_slice[data_end..].starts_with(b"\nendstream") {
        return;
    }

    let raw_xref = &obj_slice[data_start..data_end];
    let mut enc = ZlibEncoder::new(Vec::new(), Compression::best());
    if enc.write_all(raw_xref).is_err() {
        return;
    }
    let Ok(compressed) = enc.finish() else {
        return;
    };
    if compressed.len() + 19 >= raw_xref.len() {
        return;
    }

    let abs_len_start = dict_start_rel + len_kw_rel + b"/Length".len();
    let abs_len_end = dict_start_rel + digits_end;
    let mut new_tail = Vec::with_capacity(obj_slice.len() + 32);
    new_tail.extend_from_slice(&obj_slice[..abs_len_start]);
    new_tail.extend_from_slice(format!(" {}", compressed.len()).as_bytes());
    new_tail.extend_from_slice(&obj_slice[abs_len_end..dict_end_rel]);
    new_tail.extend_from_slice(b"/Filter/FlateDecode>>stream\n");
    new_tail.extend_from_slice(&compressed);
    new_tail.extend_from_slice(b"\nendstream \nendobj\n");
    new_tail.extend_from_slice(format!("\nstartxref\n{xref_start}\n%%EOF").as_bytes());

    if new_tail.len() < pdf_bytes.len().saturating_sub(xref_start) {
        pdf_bytes.truncate(xref_start);
        pdf_bytes.extend_from_slice(&new_tail);
    }
}

fn estimate_dict_serialized_len(dict: &Dictionary, depth: usize) -> usize {
    if depth > 32 {
        return 8;
    }
    let mut sum = 4usize; // << >>
    for (k, v) in dict.iter() {
        sum = sum.saturating_add(k.len() + 2 + estimate_object_serialized_len_inner(v, depth + 1));
    }
    sum
}

fn estimate_object_serialized_len_inner(obj: &Object, depth: usize) -> usize {
    if depth > 32 {
        return 8;
    }
    match obj {
        Object::Null => 4,
        Object::Boolean(_) => 5,
        Object::Integer(_) => 6,
        Object::Real(_) => 8,
        Object::Name(n) => n.len() + 2,
        Object::String(s, _) => s.len() + 3,
        Object::Reference(_) => 9,
        Object::Array(arr) => {
            let mut sum = 2usize;
            for item in arr {
                sum = sum.saturating_add(estimate_object_serialized_len_inner(item, depth + 1) + 1);
            }
            sum
        }
        Object::Dictionary(dict) => estimate_dict_serialized_len(dict, depth + 1),
        Object::Stream(stream) => estimate_dict_serialized_len(&stream.dict, depth + 1) + 36,
    }
}

fn estimate_object_serialized_len(obj: &Object) -> usize {
    estimate_object_serialized_len_inner(obj, 0)
}

/// Returns `true` for 1-bit bi-level streams (including `/ImageMask true`, `/BitsPerComponent 1`,
/// `/JBIG2Decode`, and `/CCITTFaxDecode` unless explicitly 8-bit) that are already near-optimal
/// and must never be re-encoded into larger 8-bit/24-bit raster streams.
fn is_bilevel_1bit_stream(doc: &Document, stream: &Stream) -> bool {
    if is_image_mask(doc, stream) {
        return true;
    }
    let explicit_bpc = dict_get_resolved(doc, &stream.dict, b"BitsPerComponent").and_then(obj_as_i64);
    if explicit_bpc == Some(1) {
        return true;
    }
    let filters = extract_filters(doc, &stream.dict).unwrap_or_default();
    let is_bilevel_codec = filters
        .iter()
        .any(|&f| matches!(f, b"JBIG2Decode" | b"CCITTFaxDecode"));
    is_bilevel_codec && explicit_bpc != Some(8)
}

/// Inspects a PDF in memory using the same reachability GC and hash deduplication pass as the
/// compressor, partitioning unique 1-bit/fixed streams, compressible (>=8-bit) raster images,
/// non-image `/FlateDecode` streams, and structural dictionary overhead for fast O(1) UI sizing.
pub fn inspect_pdf(input_bytes: &[u8]) -> Result<PdfSummary, PdfError> {
    let mut doc = load_unencrypted_document(input_bytes)?;
    #[cfg(test)]
    let opt_report = optimize_structure_gc_and_dedup(&mut doc, false);
    #[cfg(not(test))]
    let _ = optimize_structure_gc_and_dedup(&mut doc, false);

    let page_count = doc.get_pages().len();
    let candidate_ids = collect_candidate_image_ids(&doc);

    let mut image_count = 0usize;
    let mut compressible_image_count = 0usize;
    #[cfg(test)]
    let mut skipped_1bit_count = 0usize;
    let mut fixed_image_bytes = 0usize;

    let mut candidate_smask_ids: Vec<ObjectId> = Vec::new();
    let mut candidate_smask_seen: HashSet<ObjectId> = HashSet::new();
    let mut fixed_aux_ids: Vec<ObjectId> = Vec::new();
    let mut fixed_aux_seen: HashSet<ObjectId> = HashSet::new();
    let mut counted_ids = HashSet::new();

    let mut compressible_rgb_pixels = 0u64;
    let mut compressible_gray_pixels = 0u64;
    let mut orig_dct_image_count = 0usize;
    let mut orig_dct_bytes = 0usize;
    let mut orig_dct_rgb_pixels = 0u64;
    let mut orig_dct_gray_pixels = 0u64;
    let mut best_sample: Option<(ObjectId, u32, u32, u64)> = None;

    for id in candidate_ids {
        if !counted_ids.insert(id) {
            continue;
        }
        let Some(Object::Stream(stream)) = doc.objects.get(&id) else {
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
        let mask_ref_id = match stream.dict.get(b"Mask") {
            Ok(Object::Reference(ref_id)) => Some(*ref_id),
            _ => None,
        };

        if is_bilevel_1bit_stream(&doc, stream) {
            #[cfg(test)]
            {
                skipped_1bit_count += 1;
            }
            fixed_image_bytes = fixed_image_bytes.saturating_add(stream.content.len());
            if let Some(ref_id) = smask_ref_id {
                if fixed_aux_seen.insert(ref_id) {
                    fixed_aux_ids.push(ref_id);
                }
            }
            if let Some(ref_id) = mask_ref_id {
                if fixed_aux_seen.insert(ref_id) {
                    fixed_aux_ids.push(ref_id);
                }
            }
            continue;
        }

        let Some((w, h, is_gray, filters)) = inspect_compressible_metadata(&doc, stream) else {
            fixed_image_bytes = fixed_image_bytes.saturating_add(stream.content.len());
            if let Some(ref_id) = smask_ref_id {
                if fixed_aux_seen.insert(ref_id) {
                    fixed_aux_ids.push(ref_id);
                }
            }
            if let Some(ref_id) = mask_ref_id {
                if fixed_aux_seen.insert(ref_id) {
                    fixed_aux_ids.push(ref_id);
                }
            }
            continue;
        };

        compressible_image_count += 1;
        if let Some(ref_id) = smask_ref_id {
            if candidate_smask_seen.insert(ref_id) {
                candidate_smask_ids.push(ref_id);
            }
        }
        // Stencil `/Mask` streams are 1-bit masks and remain untouched.
        if let Some(ref_id) = mask_ref_id {
            if fixed_aux_seen.insert(ref_id) {
                fixed_aux_ids.push(ref_id);
            }
        }

        let pixels = (w as u64) * (h as u64);
        if is_gray {
            compressible_gray_pixels = compressible_gray_pixels.saturating_add(pixels);
        } else {
            compressible_rgb_pixels = compressible_rgb_pixels.saturating_add(pixels);
        }

        if filters.len() == 1 && filters[0] == b"DCTDecode" && smask_ref_id.is_none() {
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

    // Partition subsidiary `/SMask` streams: 1-bit masks stay fixed, 8-bit gray masks scale < 1.0x.
    let mut accounted_image_ids = counted_ids.clone();
    for smask_id in &candidate_smask_ids {
        if counted_ids.contains(smask_id) {
            continue;
        }
        let Some(Object::Stream(s_stream)) = doc.objects.get(smask_id) else {
            continue;
        };
        accounted_image_ids.insert(*smask_id);
        let s_orig_bytes = s_stream.content.len();
        if is_bilevel_1bit_stream(&doc, s_stream) {
            #[cfg(test)]
            {
                skipped_1bit_count += 1;
            }
            fixed_image_bytes = fixed_image_bytes.saturating_add(s_orig_bytes);
            continue;
        }
        let Some((sw, sh)) = read_image_dimensions(&doc, s_stream) else {
            fixed_image_bytes = fixed_image_bytes.saturating_add(s_orig_bytes);
            continue;
        };
        compressible_gray_pixels = compressible_gray_pixels.saturating_add((sw as u64) * (sh as u64));
    }

    for aux_id in fixed_aux_ids {
        if accounted_image_ids.contains(&aux_id) {
            continue;
        }
        if let Some(Object::Stream(aux_stream)) = doc.objects.get(&aux_id) {
            accounted_image_ids.insert(aux_id);
            #[cfg(test)]
            if is_bilevel_1bit_stream(&doc, aux_stream) {
                skipped_1bit_count += 1;
            }
            fixed_image_bytes = fixed_image_bytes.saturating_add(aux_stream.content.len());
        }
    }

    let mut dedup_flate_stream_bytes = 0usize;
    let mut other_non_image_stream_bytes = PDF_MODERN_CONTAINER_OVERHEAD_BYTES;
    let mut structural_dict_bytes = doc.objects.len().saturating_mul(18);

    for (&id, obj) in &doc.objects {
        match obj {
            Object::Stream(stream) => {
                // Streams cannot reside inside PDF object streams (/ObjStm), so count their
                // wrapper headers/dicts uncompressed at 1.0x rather than 0.35x.
                other_non_image_stream_bytes = other_non_image_stream_bytes
                    .saturating_add(estimate_dict_serialized_len(&stream.dict, 0) + 36);
                if accounted_image_ids.contains(&id) {
                    continue;
                }
                let filters = extract_filters(&doc, &stream.dict).unwrap_or_default();
                let is_meta = is_metadata_stream(&doc, stream);
                if !is_meta && filters.len() == 1 && filters[0] == b"FlateDecode" {
                    dedup_flate_stream_bytes =
                        dedup_flate_stream_bytes.saturating_add(stream.content.len());
                } else if !is_meta && filters.is_empty() && stream.allows_compression {
                    let raw = stream.content.len();
                    let est = if raw > 24 {
                        ((raw as f64) * 0.45).round() as usize
                    } else {
                        raw
                    };
                    other_non_image_stream_bytes = other_non_image_stream_bytes.saturating_add(est);
                } else {
                    other_non_image_stream_bytes =
                        other_non_image_stream_bytes.saturating_add(stream.content.len());
                }
            }
            other => {
                structural_dict_bytes =
                    structural_dict_bytes.saturating_add(estimate_object_serialized_len(other) + 12);
            }
        }
    }

    let (sample_rgb_jpeg_q50_bpp, sample_rgb_flate_bpp) =
        if let Some((sample_id, sw, sh, _)) = best_sample {
            extract_image_from_doc(&doc, sample_id, sw, sh, None)
                .map(|extracted| calibrate_sample_image(&extracted.image, sw, sh))
                .unwrap_or((DEFAULT_RGB_JPEG_Q50_BPP, DEFAULT_RGB_FLATE_BPP))
        } else {
            (DEFAULT_RGB_JPEG_Q50_BPP, DEFAULT_RGB_FLATE_BPP)
        };

    let est_flate_bytes =
        ((dedup_flate_stream_bytes as f64) * FLATE_RECOMPRESS_FACTOR).round() as usize;
    let est_struct_bytes =
        ((structural_dict_bytes as f64) * OBJSTM_XREF_COMPRESSION_FACTOR).round() as usize;
    let fixed_bytes = fixed_image_bytes
        .saturating_add(est_flate_bytes)
        .saturating_add(est_struct_bytes)
        .saturating_add(other_non_image_stream_bytes);
    let non_image_overhead_bytes = fixed_bytes;

    Ok(PdfSummary {
        page_count,
        image_count,
        compressible_image_count,
        #[cfg(test)]
        skipped_1bit_count,
        file_bytes: input_bytes.len(),
        non_image_overhead_bytes,
        #[cfg(test)]
        fixed_image_bytes,
        dedup_fixed_image_bytes: fixed_image_bytes,
        dedup_flate_stream_bytes,
        structural_dict_bytes,
        other_non_image_stream_bytes,
        #[cfg(test)]
        gc_removed_objects: opt_report.gc_removed,
        #[cfg(test)]
        dedup_merged_objects: opt_report.dedup_merged,
        fixed_bytes,
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

fn calibrate_sample_image(sample_img: &DynamicImage, orig_w: u32, orig_h: u32) -> (f64, f64) {
    let orig_pixels = (orig_w as u64) * (orig_h as u64);
    let cal_img_owned;
    let cal_img = if orig_pixels > MAX_CALIBRATION_PIXELS {
        let ratio = (MAX_CALIBRATION_PIXELS as f64 / orig_pixels as f64).sqrt();
        let nw = ((orig_w as f64) * ratio).round().max(1.0) as u32;
        let nh = ((orig_h as f64) * ratio).round().max(1.0) as u32;
        cal_img_owned = sample_img.resize_exact(nw, nh, FilterType::Triangle);
        &cal_img_owned
    } else {
        sample_img
    };

    let (cw, ch) = (cal_img.width().max(1), cal_img.height().max(1));
    let cal_pixels = (cw as f64) * (ch as f64);
    if cal_pixels <= 0.0 {
        return (DEFAULT_RGB_JPEG_Q50_BPP, DEFAULT_RGB_FLATE_BPP);
    }

    let rgb_img = DynamicImage::ImageRgb8(cal_img.to_rgb8());

    let jpeg_bpp = match encode_jpeg_stream(&rgb_img, cw, ch, 50) {
        Ok(s) => ((s.bytes.len() as f64) * 8.0 / cal_pixels).clamp(0.08, 8.0),
        Err(_) => DEFAULT_RGB_JPEG_Q50_BPP,
    };

    let flate_bpp = match encode_pdf_flate_lossless(&rgb_img, cw, ch) {
        Ok(s) => ((s.bytes.len() as f64) * 8.0 / cal_pixels).clamp(0.35, 24.0),
        Err(_) => DEFAULT_RGB_FLATE_BPP,
    };

    (jpeg_bpp, flate_bpp)
}

fn inspect_compressible_metadata<'a>(
    doc: &'a Document,
    stream: &'a Stream,
) -> Option<(u32, u32, bool, Vec<&'a [u8]>)> {
    if is_bilevel_1bit_stream(doc, stream) {
        return None;
    }
    let (w, h) = read_image_dimensions(doc, stream)?;
    if w < MIN_IMAGE_DIMENSION || h < MIN_IMAGE_DIMENSION {
        return None;
    }
    if has_unsupported_chroma_mask(doc, stream) {
        return None;
    }
    let filters = extract_filters(doc, &stream.dict).ok()?;
    let codec = validate_image_filters(&filters).ok()?;
    let is_gray = match resolve_color_space_spec(doc, stream) {
        Ok(cs) => matches!(
            cs,
            ColorSpaceSpec::Direct(BaseColorSpace::Gray)
                | ColorSpaceSpec::Indexed {
                    base: BaseColorSpace::Gray,
                    ..
                }
        ),
        Err(_) if codec == FinalImageCodec::Jpx => {
            let payload = extract_codec_payload(doc, stream, &filters, codec).ok()?;
            let settings = JpxDecodeSettings {
                resolve_palette_indices: true,
                strict: false,
                target_resolution: None,
            };
            let jpx = JpxImage::new(&payload, &settings).ok()?;
            matches!(jpx.color_space(), JpxColorSpace::Gray)
        }
        Err(_) if matches!(codec, FinalImageCodec::Jbig2 | FinalImageCodec::CcittFax) => true,
        Err(_) => return None,
    };
    Some((w, h, is_gray, filters))
}

struct ImageCompressionJob {
    id: ObjectId,
    orig_w: u32,
    orig_h: u32,
    new_w: u32,
    new_h: u32,
    target_hint: Option<(u32, u32)>,
}

enum ImageCompressionOutcome {
    Recompressed {
        id: ObjectId,
        new_w: u32,
        new_h: u32,
        encoded: EncodedPdfStream,
        keep_smask: bool,
        keep_mask: bool,
        prepared_smask: Option<(ObjectId, u32, u32, EncodedPdfStream)>,
    },
    KeptOriginal,
    SkippedUnsupported(String),
}

fn is_single_dct_filter(doc: &Document, stream: &Stream) -> bool {
    extract_filters(doc, &stream.dict)
        .is_ok_and(|filters| filters.len() == 1 && filters[0] == b"DCTDecode")
}

fn sync_all_stream_lengths(doc: &mut Document) {
    for obj in doc.objects.values_mut() {
        if let Object::Stream(stream) = obj {
            stream
                .dict
                .set("Length", Object::Integer(stream.content.len() as i64));
        }
    }
}

/// Compresses embedded images inside a PDF while leaving page boxes (`/MediaBox`, `/CropBox`)
/// and content streams (`cm` transformation matrices) untouched.
pub fn compress_pdf<F>(
    input_bytes: &[u8],
    options: &CompressionOptions,
    mut on_progress: F,
) -> Result<(Vec<u8>, CompressionStats), PdfError>
where
    F: FnMut(ProgressUpdate) + Send,
{
    let mut doc = load_unencrypted_document(input_bytes)?;

    // Priority 1 (pre-image): strip orphaned incremental objects and deduplicate identical
    // objects/streams before decoding images so duplicate images are processed only once.
    let mut pre_opt = optimize_structure_gc_and_dedup(&mut doc, true);
    let candidate_ids = collect_candidate_image_ids(&doc);

    let total_candidates = candidate_ids.len();
    let mut stats = CompressionStats {
        original_bytes: input_bytes.len(),
        total_image_objects: total_candidates,
        objects_pruned_gc: pre_opt.gc_removed,
        #[cfg(test)]
        objects_deduplicated: pre_opt.dedup_merged,
        ..Default::default()
    };
    #[cfg(not(test))]
    let _ = pre_opt.dedup_merged;

    let scale = if options.scale.is_finite() {
        options.scale.clamp(0.10, 1.00)
    } else {
        1.00
    };
    let scaling_requested = (scale - 1.0).abs() >= 1e-4;
    let quality = options.quality.clamp(1, 100);
    let filter_type = options.resize_filter.to_image_filter();
    let passthrough_dct =
        options.format == ImageFormatChoice::Jpeg && !scaling_requested && quality == 100;

    // Phase 1 (sequential, fast): filter candidates and perform Q100/1.00x /DCTDecode passthrough.
    let mut jobs = Vec::with_capacity(total_candidates);
    for id in candidate_ids {
        let (is_1bit, dims, has_bad_chroma_mask, is_single_dct) = match doc.objects.get(&id) {
            Some(Object::Stream(stream)) => (
                is_bilevel_1bit_stream(&doc, stream),
                read_image_dimensions(&doc, stream),
                has_unsupported_chroma_mask(&doc, stream),
                passthrough_dct && is_single_dct_filter(&doc, stream),
            ),
            _ => continue,
        };

        if is_1bit {
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

        if has_bad_chroma_mask {
            stats.images_skipped_unsupported += 1;
            push_unique_reason(
                &mut stats.skipped_reasons,
                "Skipped an image using an explicit /Mask chroma-key array.".to_string(),
            );
            continue;
        }

        if is_single_dct {
            stats.images_kept_original += 1;
            continue;
        }

        let (new_w, new_h) = if !scaling_requested {
            (orig_w, orig_h)
        } else {
            (
                ((orig_w as f32) * scale).round().max(1.0) as u32,
                ((orig_h as f32) * scale).round().max(1.0) as u32,
            )
        };
        let target_hint = if scaling_requested {
            Some((new_w.saturating_mul(2).max(1), new_h.saturating_mul(2).max(1)))
        } else {
            None
        };

        jobs.push(ImageCompressionJob {
            id,
            orig_w,
            orig_h,
            new_w,
            new_h,
            target_hint,
        });
    }

    // Phase 2 (parallel across CPU cores): decode, resize, and re-encode independent XObjects.
    let pre_handled = stats.images_skipped_small_or_mask
        + stats.images_skipped_unsupported
        + stats.images_kept_original;
    let total_progress = total_candidates.max(1);
    if jobs.is_empty() {
        on_progress(ProgressUpdate {
            current: total_progress,
            total: total_progress,
        });
    } else {
        on_progress(ProgressUpdate {
            current: pre_handled,
            total: total_progress,
        });
    }

    let processed_count = AtomicUsize::new(0);
    let progress_lock = Mutex::new(&mut on_progress);

    let outcomes: Vec<Result<ImageCompressionOutcome, PdfError>> = {
        let doc_ref = &doc;
        jobs.par_iter()
            .map(|job| {
                let outcome = match extract_image_from_doc(
                    doc_ref,
                    job.id,
                    job.orig_w,
                    job.orig_h,
                    job.target_hint,
                ) {
                    Ok(extracted) => {
                        let prepared_smask = if let Some(smask_id) = extracted.smask_id {
                            prepare_scaled_smask(doc_ref, smask_id, scale, filter_type)?
                        } else {
                            None
                        };

                        let resized = if extracted.image.width() == job.new_w
                            && extracted.image.height() == job.new_h
                        {
                            extracted.image
                        } else {
                            extracted.image.resize_exact(job.new_w, job.new_h, filter_type)
                        };

                        let encoded =
                            encode_for_pdf(&resized, job.new_w, job.new_h, options.format, quality)?;

                        if encoded.bytes.len() >= extracted.orig_raw_len {
                            ImageCompressionOutcome::KeptOriginal
                        } else {
                            ImageCompressionOutcome::Recompressed {
                                id: job.id,
                                new_w: job.new_w,
                                new_h: job.new_h,
                                encoded,
                                keep_smask: extracted.smask_id.is_some(),
                                keep_mask: extracted.stencil_mask_id.is_some(),
                                prepared_smask,
                            }
                        }
                    }
                    Err(SkipReason::Unsupported(reason)) => {
                        ImageCompressionOutcome::SkippedUnsupported(reason)
                    }
                };

                let done = pre_handled + processed_count.fetch_add(1, Ordering::Relaxed) + 1;
                if let Ok(mut cb) = progress_lock.try_lock() {
                    cb(ProgressUpdate {
                        current: done.min(total_progress),
                        total: total_progress,
                    });
                }

                Ok(outcome)
            })
            .collect()
    };

    // Phase 3 (sequential deterministic apply): mutate lopdf objects in document order.
    let mut processed_smasks: HashSet<ObjectId> = HashSet::new();
    for res in outcomes {
        match res? {
            ImageCompressionOutcome::KeptOriginal => {
                stats.images_kept_original += 1;
            }
            ImageCompressionOutcome::SkippedUnsupported(reason) => {
                stats.images_skipped_unsupported += 1;
                push_unique_reason(&mut stats.skipped_reasons, reason);
            }
            ImageCompressionOutcome::Recompressed {
                id,
                new_w,
                new_h,
                encoded,
                keep_smask,
                keep_mask,
                prepared_smask,
            } => {
                if let Some((smask_id, sw, sh, smask_encoded)) = prepared_smask {
                    if processed_smasks.insert(smask_id) {
                        pre_opt.cached_flate_decoded.remove(&smask_id);
                        if let Some(Object::Stream(s_stream)) = doc.objects.get_mut(&smask_id) {
                            apply_encoded_image_to_stream(
                                s_stream,
                                sw,
                                sh,
                                smask_encoded,
                                false,
                                false,
                            );
                        }
                    }
                }
                pre_opt.cached_flate_decoded.remove(&id);
                if let Some(Object::Stream(stream)) = doc.objects.get_mut(&id) {
                    apply_encoded_image_to_stream(
                        stream,
                        new_w,
                        new_h,
                        encoded,
                        keep_smask,
                        keep_mask,
                    );
                    stats.images_recompressed += 1;
                }
            }
        }
    }

    on_progress(ProgressUpdate {
        current: total_progress,
        total: total_progress,
    });

    // Priority 3: high-ratio lossless Flate re-compression of content streams, fonts, CMaps,
    // ICC profiles, and any uncompressed non-metadata streams (guarded strictly so nothing bloats).
    stats.flate_streams_recompressed =
        recompress_flate_and_raw_streams(&mut doc, pre_opt.cached_flate_decoded);

    // Post-compression GC pass to prune any newly orphaned auxiliary streams.
    let post_gc = garbage_collect_unreachable_objects(&mut doc);
    stats.objects_pruned_gc = stats.objects_pruned_gc.saturating_add(post_gc);

    // Priority 2: PDF 1.5+ ObjStm + binary XRef stream serialization.
    ensure_min_pdf_version(&mut doc, 1, 5);
    normalize_generations_to_zero(&mut doc);
    doc.renumber_objects();
    sync_all_stream_lengths(&mut doc);

    let save_options = SaveOptions::builder()
        .use_object_streams(true)
        .use_xref_streams(true)
        .compression_level(9)
        .max_objects_per_stream(200)
        .build();

    let mut out_bytes = Vec::with_capacity(input_bytes.len());
    doc.save_with_options(&mut out_bytes, save_options)
        .map_err(|e| PdfError::Save(e.to_string()))?;
    compress_trailing_xref_stream(&mut out_bytes);

    stats.compressed_bytes = out_bytes.len();
    Ok((out_bytes, stats))
}

fn load_unencrypted_document(input_bytes: &[u8]) -> Result<Document, PdfError> {
    if input_bytes.is_empty() {
        return Err(PdfError::Parse("The PDF file is empty (0 bytes).".to_string()));
    }
    let doc = Document::load_mem(input_bytes).map_err(|e| {
        let msg = e.to_string();
        let lower = msg.to_ascii_lowercase();
        if lower.contains("encrypt") || lower.contains("decrypt") || lower.contains("password") {
            PdfError::Encrypted
        } else {
            PdfError::Parse(msg)
        }
    })?;
    if doc.is_encrypted() || doc.trailer.get(b"Encrypt").is_ok() {
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

fn extract_filters<'a>(
    doc: &'a Document,
    dict: &'a Dictionary,
) -> Result<Vec<&'a [u8]>, SkipReason> {
    let Some(filter_obj) = dict_get_resolved(doc, dict, b"Filter") else {
        return Ok(Vec::new());
    };
    match filter_obj {
        Object::Name(name) => Ok(vec![name.as_slice()]),
        Object::Array(arr) => {
            let mut out = Vec::with_capacity(arr.len());
            for item in arr {
                let resolved = resolve_obj(doc, item).ok_or_else(|| {
                    SkipReason::Unsupported("Unresolvable /Filter entry in image stream.".into())
                })?;
                let name = obj_as_name(resolved).ok_or_else(|| {
                    SkipReason::Unsupported("Invalid non-name /Filter entry in image stream.".into())
                })?;
                out.push(name);
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

fn dict_entry_needs_clone_resolution(dict: &Dictionary, key: &[u8]) -> bool {
    match dict.get(key) {
        Ok(Object::Reference(_)) | Ok(Object::Null) => true,
        Ok(Object::Array(arr)) => {
            arr.is_empty()
                || arr.iter().any(|o| matches!(o, Object::Reference(_)))
                || arr.iter().all(|o| matches!(o, Object::Null))
        }
        Ok(_) | Err(_) => false,
    }
}

fn decompress_stream_with_resolved_dict(
    doc: &Document,
    stream: &Stream,
    strip_last_filter: bool,
) -> Result<Vec<u8>, SkipReason> {
    if !strip_last_filter {
        if stream.dict.get(b"Filter").is_err() {
            return Ok(stream.content.clone());
        }
        if !dict_entry_needs_clone_resolution(&stream.dict, b"Filter")
            && !dict_entry_needs_clone_resolution(&stream.dict, b"DecodeParms")
        {
            return stream
                .decompressed_content_with_limit(MAX_DECOMPRESSED_BYTES)
                .map_err(|e| SkipReason::Unsupported(format!("Stream decompression skipped: {e}")));
        }
    }

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
                    if arr.iter().all(|o| matches!(o, Object::Null)) {
                        cloned.dict.remove(b"DecodeParms");
                    } else if arr.len() == 1 {
                        cloned.dict.set("DecodeParms", arr.remove(0));
                    } else {
                        cloned.dict.set("DecodeParms", Object::Array(arr));
                    }
                }
                _ => {
                    cloned.dict.remove(b"DecodeParms");
                }
            }
        } else if matches!(resolved, Object::Null)
            || matches!(&resolved, Object::Array(arr) if arr.iter().all(|o| matches!(o, Object::Null)))
        {
            cloned.dict.remove(b"DecodeParms");
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

fn has_unsupported_chroma_mask(doc: &Document, stream: &Stream) -> bool {
    let Ok(mask_obj) = stream.dict.get(b"Mask") else {
        return false;
    };
    match resolve_obj(doc, mask_obj) {
        Some(Object::Stream(_)) => false,
        _ => true,
    }
}

fn validate_image_filters(filters: &[&[u8]]) -> Result<FinalImageCodec, SkipReason> {
    let (codec, prefix_limit) = match filters.last().copied() {
        Some(b"DCTDecode") => (FinalImageCodec::Dct, filters.len() - 1),
        Some(b"JPXDecode") => (FinalImageCodec::Jpx, filters.len() - 1),
        Some(b"JBIG2Decode") => (FinalImageCodec::Jbig2, filters.len() - 1),
        Some(b"CCITTFaxDecode") => (FinalImageCodec::CcittFax, filters.len() - 1),
        _ => (FinalImageCodec::RawSamples, filters.len()),
    };

    for &f in &filters[..prefix_limit] {
        if !matches!(
            f,
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

    Ok(codec)
}

fn extract_codec_payload<'a>(
    doc: &'a Document,
    stream: &'a Stream,
    filters: &[&[u8]],
    codec: FinalImageCodec,
) -> Result<Cow<'a, [u8]>, SkipReason> {
    if codec == FinalImageCodec::RawSamples {
        if filters.is_empty() {
            Ok(Cow::Borrowed(&stream.content))
        } else {
            decompress_stream_with_resolved_dict(doc, stream, false).map(Cow::Owned)
        }
    } else if filters.len() == 1 {
        Ok(Cow::Borrowed(&stream.content))
    } else {
        decompress_stream_with_resolved_dict(doc, stream, true).map(Cow::Owned)
    }
}

fn extract_codec_decode_parms<'a>(
    doc: &'a Document,
    dict: &'a Dictionary,
) -> Option<&'a Dictionary> {
    let parms_obj = dict_get_resolved(doc, dict, b"DecodeParms")?;
    match parms_obj {
        Object::Dictionary(d) => Some(d),
        Object::Array(arr) => arr
            .last()
            .and_then(|item| resolve_obj(doc, item))
            .and_then(|obj| match obj {
                Object::Dictionary(d) => Some(d),
                _ => None,
            }),
        _ => None,
    }
}

fn decode_stream_to_image(
    doc: &Document,
    stream: &Stream,
    width: u32,
    height: u32,
    force_gray: bool,
    is_mask: bool,
    target_resolution: Option<(u32, u32)>,
) -> Result<DynamicImage, SkipReason> {
    let filters = extract_filters(doc, &stream.dict)?;
    let codec = validate_image_filters(&filters)?;
    let payload = extract_codec_payload(doc, stream, &filters, codec)?;

    let img = match codec {
        FinalImageCodec::Dct => decode_dct_image(doc, stream, &payload)?,
        FinalImageCodec::Jpx => decode_jpx_image(doc, stream, &payload, target_resolution)?,
        FinalImageCodec::Jbig2 => {
            let packed = decode_jbig2_to_1bit_packed(doc, stream, &payload, width, height)?;
            decode_bilevel_packed(doc, stream, &packed, width, height, force_gray || is_mask, is_mask)?
        }
        FinalImageCodec::CcittFax => {
            let packed = decode_ccitt_to_1bit_packed(doc, stream, &payload, width, height)?;
            decode_bilevel_packed(doc, stream, &packed, width, height, force_gray || is_mask, is_mask)?
        }
        FinalImageCodec::RawSamples => {
            if force_gray || is_mask {
                let bpc = read_bits_per_component_with_default(doc, stream, if is_mask { 1 } else { 8 });
                let decode_arr = read_decode_array(doc, stream);
                decode_uncompressed_to_image(
                    &payload,
                    width,
                    height,
                    bpc,
                    &ColorSpaceSpec::Direct(BaseColorSpace::Gray),
                    decode_arr.as_deref(),
                )?
            } else {
                decode_raw_pdf_samples(doc, stream, &payload, width, height)?
            }
        }
    };

    let final_img = if force_gray || is_mask {
        DynamicImage::ImageLuma8(img.into_luma8())
    } else {
        img
    };

    Ok(final_img)
}

fn extract_image_from_doc(
    doc: &Document,
    id: ObjectId,
    orig_w: u32,
    orig_h: u32,
    target_resolution: Option<(u32, u32)>,
) -> Result<ExtractedImage, SkipReason> {
    let stream = match doc.objects.get(&id) {
        Some(Object::Stream(s)) => s,
        _ => {
            return Err(SkipReason::Unsupported(
                "Image object is not a stream.".into(),
            ))
        }
    };

    if has_unsupported_chroma_mask(doc, stream) {
        return Err(SkipReason::Unsupported(
            "Skipped an image using an explicit /Mask chroma-key array.".into(),
        ));
    }

    let orig_raw_len = stream.content.len();
    let base_img =
        decode_stream_to_image(doc, stream, orig_w, orig_h, false, false, target_resolution)?;

    let smask_id = match stream.dict.get(b"SMask") {
        Ok(Object::Reference(ref_id)) if doc.objects.get(ref_id).is_some() => Some(*ref_id),
        Ok(_) => {
            return Err(SkipReason::Unsupported(
                "Skipped an image with a non-indirect /SMask reference.".into(),
            ));
        }
        Err(_) => None,
    };

    // Validate explicit non-1-bit /SMask stream is decodable cleanly; 1-bit /SMask stays untouched.
    if let Some(sid) = smask_id {
        if let Some(Object::Stream(s_stream)) = doc.objects.get(&sid) {
            if !is_bilevel_1bit_stream(doc, s_stream) {
                let (sw, sh) = read_image_dimensions(doc, s_stream).ok_or_else(|| {
                    SkipReason::Unsupported("Skipped an image with an invalid /SMask size.".into())
                })?;
                let s_filters = extract_filters(doc, &s_stream.dict)?;
                validate_image_filters(&s_filters)?;
                let _ = (sw, sh);
            }
        } else {
            return Err(SkipReason::Unsupported(
                "Skipped an image with a missing /SMask stream.".into(),
            ));
        }
    }

    let stencil_mask_id = match stream.dict.get(b"Mask") {
        Ok(Object::Reference(ref_id)) => Some(*ref_id),
        _ => None,
    };

    Ok(ExtractedImage {
        image: base_img,
        orig_raw_len,
        smask_id,
        stencil_mask_id,
    })
}

/// Scales an 8-bit grayscale `/SMask` stream when downscaling (`scale < 1.0`) or uncompressed,
/// preserving `/ColorSpace /DeviceGray`, `/BitsPerComponent 8`, and `/Filter /FlateDecode`.
/// Leaves 1-bit `/SMask` streams completely untouched and guards compressed `/SMask`s against bloat.
fn prepare_scaled_smask(
    doc: &Document,
    smask_id: ObjectId,
    scale: f32,
    filter_type: FilterType,
) -> Result<Option<(ObjectId, u32, u32, EncodedPdfStream)>, PdfError> {
    let Some(Object::Stream(smask_stream)) = doc.objects.get(&smask_id) else {
        return Ok(None);
    };
    if is_bilevel_1bit_stream(doc, smask_stream) {
        return Ok(None);
    }
    let Some((sw, sh)) = read_image_dimensions(doc, smask_stream) else {
        return Ok(None);
    };
    let orig_len = smask_stream.content.len();

    let filters = extract_filters(doc, &smask_stream.dict).unwrap_or_default();
    let s_is_compressed = !filters.is_empty();
    let scaling_requested = (scale - 1.0).abs() >= 1e-4;
    if !scaling_requested && s_is_compressed {
        // Keep already-compressed 8-bit /SMask streams untouched at 1.0x.
        return Ok(None);
    }

    let nw = ((sw as f32) * scale).round().max(1.0) as u32;
    let nh = ((sh as f32) * scale).round().max(1.0) as u32;
    let target_hint = if scaling_requested {
        Some((nw.saturating_mul(2).max(1), nh.saturating_mul(2).max(1)))
    } else {
        None
    };

    let alpha_img =
        decode_stream_to_image(doc, smask_stream, sw, sh, true, false, target_hint).map_err(|e| {
            match e {
                SkipReason::Unsupported(msg) => PdfError::ImageEncode(msg),
            }
        })?;
    let resized = if alpha_img.width() == nw && alpha_img.height() == nh {
        alpha_img
    } else {
        alpha_img.resize_exact(nw, nh, filter_type)
    };

    let luma = DynamicImage::ImageLuma8(resized.into_luma8());
    let encoded = encode_pdf_flate_lossless(&luma, nw, nh)?;
    if s_is_compressed && encoded.bytes.len() >= orig_len {
        return Ok(None);
    }
    Ok(Some((smask_id, nw, nh, encoded)))
}

fn decode_dct_image(
    doc: &Document,
    stream: &Stream,
    jpeg_bytes: &[u8],
) -> Result<DynamicImage, SkipReason> {
    let mut img = decode_jpeg_to_image(jpeg_bytes)
        .map_err(|e| SkipReason::Unsupported(format!("Skipped unreadable JPEG stream: {e}")))?;

    // Check if an explicit inverted /Decode array exists on grayscale/RGB JPEGs.
    apply_decode_array_to_dynamic_image(doc, stream, &mut img);

    Ok(img)
}

fn decode_jpx_image(
    doc: &Document,
    stream: &Stream,
    jpx_bytes: &[u8],
    target_resolution: Option<(u32, u32)>,
) -> Result<DynamicImage, SkipReason> {
    let pdf_cs = resolve_color_space_spec(doc, stream).ok();
    let is_pdf_indexed = matches!(&pdf_cs, Some(ColorSpaceSpec::Indexed { .. }));

    let settings = JpxDecodeSettings {
        resolve_palette_indices: !is_pdf_indexed,
        strict: false,
        target_resolution: if is_pdf_indexed {
            None
        } else {
            target_resolution
        },
    };

    let jpx = JpxImage::new(jpx_bytes, &settings).map_err(|e| {
        SkipReason::Unsupported(format!("Skipped unreadable JPEG2000 stream: {e:?}"))
    })?;

    if let Some(ColorSpaceSpec::Indexed {
        base,
        hival,
        lookup,
    }) = pdf_cs
    {
        let mut ctx = JpxDecoderContext::default();
        let decoded = jpx.decode(&mut ctx).map_err(|e| {
            SkipReason::Unsupported(format!("Skipped JPEG2000 indexed decode error: {e:?}"))
        })?;
        let raw_indices = decoded.data_u8();
        let w = jpx.width();
        let h = jpx.height();
        let mut img = expand_indexed_samples(&raw_indices, w, h, &base, hival, &lookup)?;
        apply_decode_array_to_dynamic_image(doc, stream, &mut img);
        return Ok(img);
    }

    let jpx_w = jpx.width();
    let jpx_h = jpx.height();
    let jpx_cs = jpx.color_space().clone();
    let mut img = match DynamicImage::from_decoder(jpx) {
        Ok(dyn_img) => match dyn_img {
            DynamicImage::ImageLuma8(_) | DynamicImage::ImageRgb8(_) => dyn_img,
            DynamicImage::ImageLumaA8(_) | DynamicImage::ImageLuma16(_) | DynamicImage::ImageLumaA16(_) => {
                DynamicImage::ImageLuma8(dyn_img.into_luma8())
            }
            _ => DynamicImage::ImageRgb8(dyn_img.into_rgb8()),
        },
        Err(_) => {
            // Fallback for Unknown JPX channel layouts using PDF /ColorSpace hints.
            let jpx_retry = JpxImage::new(jpx_bytes, &settings).map_err(|e| {
                SkipReason::Unsupported(format!("Skipped unreadable JPEG2000 stream: {e:?}"))
            })?;
            let mut ctx = JpxDecoderContext::default();
            let decoded = jpx_retry.decode(&mut ctx).map_err(|e| {
                SkipReason::Unsupported(format!("Skipped JPEG2000 decode error: {e:?}"))
            })?;
            let raw = decoded.data_u8();
            let expected_comps = jpx_cs.num_channels() as usize;
            let spec = match pdf_cs {
                Some(cs) => cs,
                None => match expected_comps {
                    1 => ColorSpaceSpec::Direct(BaseColorSpace::Gray),
                    4 => ColorSpaceSpec::Direct(BaseColorSpace::Cmyk),
                    _ => ColorSpaceSpec::Direct(BaseColorSpace::Rgb),
                },
            };
            decode_uncompressed_to_image(&raw, jpx_w, jpx_h, 8, &spec, None)?
        }
    };

    apply_decode_array_to_dynamic_image(doc, stream, &mut img);
    Ok(img)
}

fn decode_jbig2_to_1bit_packed(
    doc: &Document,
    stream: &Stream,
    jbig2_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, SkipReason> {
    let globals_bytes: Option<Cow<'_, [u8]>> = extract_codec_decode_parms(doc, &stream.dict)
        .and_then(|parms| dict_get_resolved(doc, parms, b"JBIG2Globals"))
        .and_then(|obj| match obj {
            Object::Stream(s) => {
                if s.dict.get(b"Filter").is_ok() {
                    decompress_stream_with_resolved_dict(doc, s, false)
                        .ok()
                        .map(Cow::Owned)
                } else {
                    Some(Cow::Borrowed(s.content.as_slice()))
                }
            }
            _ => None,
        });

    let settings = Jbig2DecodeSettings { strict: false };
    let globals = match globals_bytes.as_deref() {
        Some(gb) => Some(Jbig2Globals::new(gb, &settings).map_err(|e| {
            SkipReason::Unsupported(format!("Skipped invalid JBIG2 globals: {e:?}"))
        })?),
        None => None,
    };

    struct Jbig2Packer {
        out: Vec<u8>,
        out_w: usize,
        out_h: usize,
        row_bytes: usize,
        row: usize,
        col: usize,
    }

    impl hayro_jbig2::Decoder for Jbig2Packer {
        fn push_pixel(&mut self, black: bool) {
            // In PDF JBIG2 streams (`DeviceGray` / `ImageMask`), `black == false` maps to sample bit 1
            // and `black == true` maps to sample bit 0 (verified on MRC JBIG2 SMask streams).
            if self.row < self.out_h && self.col < self.out_w && !black {
                self.out[self.row * self.row_bytes + (self.col >> 3)] |= 0x80 >> (self.col & 7);
            }
            self.col += 1;
        }

        fn push_pixel_chunk(&mut self, black: bool, chunk_count: u32) {
            let count = (chunk_count as usize) * 8;
            if self.row < self.out_h {
                if !black {
                    if self.col % 8 == 0 && self.col + count <= self.out_w {
                        let start = self.row * self.row_bytes + (self.col >> 3);
                        let bytes = chunk_count as usize;
                        self.out[start..start + bytes].fill(0xFF);
                    } else {
                        let end = (self.col + count).min(self.out_w);
                        let row_off = self.row * self.row_bytes;
                        for x in self.col..end {
                            self.out[row_off + (x >> 3)] |= 0x80 >> (x & 7);
                        }
                    }
                }
            }
            self.col += count;
        }

        fn next_line(&mut self) {
            self.row += 1;
            self.col = 0;
        }
    }

    let row_bytes = (width as usize).div_ceil(8);
    let mut packer = Jbig2Packer {
        out: vec![0u8; row_bytes * (height as usize)],
        out_w: width as usize,
        out_h: height as usize,
        row_bytes,
        row: 0,
        col: 0,
    };

    if let Ok(page) = Jbig2Page::new_embedded(jbig2_bytes, globals.as_ref(), &settings) {
        page.decode(&mut packer).map_err(|e| {
            SkipReason::Unsupported(format!("Skipped unreadable JBIG2 stream: {e:?}"))
        })?;
        return Ok(packer.out);
    }

    let file = Jbig2File::new(jbig2_bytes, &settings).map_err(|e| {
        SkipReason::Unsupported(format!("Skipped unreadable JBIG2 stream: {e:?}"))
    })?;
    let page = file.page(0).ok_or_else(|| {
        SkipReason::Unsupported("Skipped empty JBIG2 file stream.".into())
    })?;
    page.decode(&mut packer).map_err(|e| {
        SkipReason::Unsupported(format!("Skipped unreadable JBIG2 page: {e:?}"))
    })?;
    Ok(packer.out)
}

fn decode_ccitt_to_1bit_packed(
    doc: &Document,
    stream: &Stream,
    ccitt_bytes: &[u8],
    width: u32,
    height: u32,
) -> Result<Vec<u8>, SkipReason> {
    let parms = extract_codec_decode_parms(doc, &stream.dict);
    let parm_i64 = |key: &[u8]| parms.and_then(|d| dict_get_resolved(doc, d, key)).and_then(obj_as_i64);
    let parm_bool = |key: &[u8]| parms.and_then(|d| dict_get_resolved(doc, d, key)).and_then(obj_as_bool);

    let k = parm_i64(b"K").unwrap_or(0);
    let encoding = if k < 0 {
        EncodingMode::Group4
    } else if k == 0 {
        EncodingMode::Group3_1D
    } else {
        EncodingMode::Group3_2D
    };

    let explicit_columns = parm_i64(b"Columns").and_then(|v| u32::try_from(v).ok()).filter(|&v| v > 0);
    let rows_parm = parm_i64(b"Rows")
        .and_then(|v| u32::try_from(v).ok())
        .filter(|&v| v > 0)
        .unwrap_or(height);
    let end_of_line = parm_bool(b"EndOfLine").unwrap_or(false);
    let rows_are_byte_aligned = parm_bool(b"EncodedByteAlign").unwrap_or(false);
    let end_of_block = parm_bool(b"EndOfBlock").unwrap_or(true);
    let black_is_1 = parm_bool(b"BlackIs1").unwrap_or(false);

    struct CcittPacker {
        out: Vec<u8>,
        out_w: usize,
        out_h: usize,
        row_bytes: usize,
        row: usize,
    }

    impl hayro_ccitt::Decoder for CcittPacker {
        fn push_row(&mut self, changes: &[u32]) {
            if self.row >= self.out_h {
                self.row += 1;
                return;
            }
            let row_off = self.row * self.row_bytes;
            // Each CCITT row starts white (bit 0); odd intervals `changes[0]..changes[1]` are black (bit 1).
            let mut idx = 0usize;
            while idx < changes.len() {
                let start = (changes[idx] as usize).min(self.out_w);
                let end = changes
                    .get(idx + 1)
                    .map(|&v| v as usize)
                    .unwrap_or(self.out_w)
                    .min(self.out_w);
                for x in start..end {
                    self.out[row_off + (x >> 3)] |= 0x80 >> (x & 7);
                }
                idx += 2;
            }
            self.row += 1;
        }
    }

    let run_decode = |columns: u32| -> Result<(Vec<u8>, usize), SkipReason> {
        let settings = CcittDecodeSettings {
            columns,
            rows: rows_parm,
            end_of_block,
            end_of_line,
            rows_are_byte_aligned,
            encoding,
            strict: false,
        };
        let mut ctx = CcittDecoderContext::new(settings);
        let row_bytes = (width as usize).div_ceil(8);
        let mut packer = CcittPacker {
            out: vec![0u8; row_bytes * (height as usize)],
            out_w: width as usize,
            out_h: height as usize,
            row_bytes,
            row: 0,
        };
        let decode_res = hayro_ccitt::decode(ccitt_bytes, &mut packer, &mut ctx);
        if decode_res.is_err() && packer.row == 0 {
            return Err(SkipReason::Unsupported(
                "Skipped unreadable CCITTFaxDecode stream.".into(),
            ));
        }
        Ok((packer.out, packer.row))
    };

    let (mut packed, _) = match explicit_columns {
        Some(cols) => run_decode(cols)?,
        None => {
            // PDF default /Columns is 1728, though some producers omit it when width != 1728.
            match run_decode(1728) {
                Ok((buf, rows)) if rows >= (height as usize) || width == 1728 => (buf, rows),
                _ => run_decode(width)?,
            }
        }
    };

    // PDF default (BlackIs1 = false): CCITT white -> PDF sample bit 1, CCITT black -> PDF sample bit 0.
    if !black_is_1 {
        for b in &mut packed {
            *b = !*b;
        }
    }

    Ok(packed)
}

fn decode_bilevel_packed(
    doc: &Document,
    stream: &Stream,
    packed: &[u8],
    width: u32,
    height: u32,
    force_gray: bool,
    is_mask: bool,
) -> Result<DynamicImage, SkipReason> {
    let cs = if force_gray || is_mask {
        ColorSpaceSpec::Direct(BaseColorSpace::Gray)
    } else {
        resolve_color_space_spec(doc, stream)
            .unwrap_or(ColorSpaceSpec::Direct(BaseColorSpace::Gray))
    };
    let decode_arr = read_decode_array(doc, stream);
    decode_uncompressed_to_image(packed, width, height, 1, &cs, decode_arr.as_deref())
}

fn apply_decode_array_to_dynamic_image(doc: &Document, stream: &Stream, img: &mut DynamicImage) {
    let Some(decode_arr) = read_decode_array(doc, stream) else {
        return;
    };
    match img {
        DynamicImage::ImageLuma8(gray) => {
            apply_decode_array_u8(gray.as_mut(), 1, &decode_arr);
        }
        DynamicImage::ImageRgb8(rgb) => {
            apply_decode_array_u8(rgb.as_mut(), 3, &decode_arr);
        }
        _ => {}
    }
}

fn read_bits_per_component(doc: &Document, stream: &Stream) -> u8 {
    read_bits_per_component_with_default(doc, stream, 8)
}

fn read_bits_per_component_with_default(doc: &Document, stream: &Stream, default: u8) -> u8 {
    dict_get_resolved(doc, &stream.dict, b"BitsPerComponent")
        .and_then(obj_as_i64)
        .and_then(|v| u8::try_from(v).ok())
        .unwrap_or(default)
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

fn resolve_color_space_spec<'a>(
    doc: &'a Document,
    stream: &'a Stream,
) -> Result<ColorSpaceSpec<'a>, SkipReason> {
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
                let lookup: Cow<'a, [u8]> = match lookup_obj {
                    Object::String(bytes, _) => Cow::Borrowed(bytes.as_slice()),
                    Object::Stream(s) => {
                        if s.dict.get(b"Filter").is_ok() {
                            Cow::Owned(decompress_stream_with_resolved_dict(doc, s, false)?)
                        } else {
                            Cow::Borrowed(s.content.as_slice())
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
    cs: &ColorSpaceSpec<'_>,
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
            expand_indexed_samples(&indices, width, height, base, *hival, lookup)
        }
        ColorSpaceSpec::Direct(base) => {
            let comps = base.components();
            let mut samples = unpack_samples_to_u8(raw, width as usize, height as usize, comps, bpc)?;
            if let Some(dec) = decode_arr {
                apply_decode_array_u8(samples.to_mut(), comps, dec);
            }

            match base {
                BaseColorSpace::Gray => {
                    let img = GrayImage::from_raw(width, height, samples.into_owned())
                        .ok_or_else(|| {
                            SkipReason::Unsupported(
                                "Failed to assemble Grayscale pixel buffer.".into(),
                            )
                        })?;
                    Ok(DynamicImage::ImageLuma8(img))
                }
                BaseColorSpace::Rgb => {
                    let img = RgbImage::from_raw(width, height, samples.into_owned()).ok_or_else(
                        || SkipReason::Unsupported("Failed to assemble RGB pixel buffer.".into()),
                    )?;
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

fn expand_indexed_samples(
    indices: &[u8],
    width: u32,
    height: u32,
    base: &BaseColorSpace,
    hival: usize,
    lookup: &[u8],
) -> Result<DynamicImage, SkipReason> {
    let base_comps = base.components();
    let needed_table_len = (hival + 1) * base_comps;
    if lookup.len() < needed_table_len {
        return Err(SkipReason::Unsupported(
            "Skipped /Indexed image with truncated palette table.".into(),
        ));
    }
    let pixel_count = (width as usize) * (height as usize);
    if indices.len() < pixel_count {
        return Err(SkipReason::Unsupported(
            "Skipped /Indexed image with truncated index samples.".into(),
        ));
    }
    match base {
        BaseColorSpace::Gray => {
            let mut out = Vec::with_capacity(pixel_count);
            for &idx_val in &indices[..pixel_count] {
                let idx = (idx_val as usize).min(hival);
                out.push(lookup[idx]);
            }
            let img = GrayImage::from_raw(width, height, out).ok_or_else(|| {
                SkipReason::Unsupported("Failed to build indexed grayscale buffer.".into())
            })?;
            Ok(DynamicImage::ImageLuma8(img))
        }
        BaseColorSpace::Rgb => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for &idx_val in &indices[..pixel_count] {
                let idx = (idx_val as usize).min(hival) * 3;
                out.extend_from_slice(&lookup[idx..idx + 3]);
            }
            let img = RgbImage::from_raw(width, height, out).ok_or_else(|| {
                SkipReason::Unsupported("Failed to build indexed RGB buffer.".into())
            })?;
            Ok(DynamicImage::ImageRgb8(img))
        }
        BaseColorSpace::Cmyk => {
            let mut out = Vec::with_capacity(pixel_count * 3);
            for &idx_val in &indices[..pixel_count] {
                let idx = (idx_val as usize).min(hival) * 4;
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
fn unpack_samples_per_row<'a>(
    raw: &'a [u8],
    width: usize,
    height: usize,
    components: usize,
    bpc: u8,
) -> Result<Cow<'a, [u8]>, SkipReason> {
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
        return Ok(Cow::Borrowed(&raw[..total_samples]));
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
    Ok(Cow::Owned(out))
}

/// Unpacks raw PDF image samples (1, 2, 4, 8, or 16 bpc) and scales each sample into 0..=255.
fn unpack_samples_to_u8<'a>(
    raw: &'a [u8],
    width: usize,
    height: usize,
    components: usize,
    bpc: u8,
) -> Result<Cow<'a, [u8]>, SkipReason> {
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
        return Ok(Cow::Owned(out));
    }

    let mut out = unpack_samples_per_row(raw, width, height, components, bpc)?;
    if bpc < 8 {
        let max_val = ((1u16 << bpc) - 1) as u16;
        for v in out.to_mut() {
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
        ImageFormatChoice::Jpeg => encode_jpeg_stream(image, width, height, quality),
        ImageFormatChoice::Jpeg2000 => encode_jpx_stream(image, width, height, quality),
        ImageFormatChoice::Png => encode_pdf_flate_lossless(image, width, height),
    }
}

fn image_raw_luma_or_rgb(image: &DynamicImage) -> (Cow<'_, [u8]>, u8, &'static [u8]) {
    match image {
        DynamicImage::ImageLuma8(gray) => (
            Cow::Borrowed(gray.as_raw().as_slice()),
            1,
            b"DeviceGray".as_slice(),
        ),
        DynamicImage::ImageRgb8(rgb) => (
            Cow::Borrowed(rgb.as_raw().as_slice()),
            3,
            b"DeviceRGB".as_slice(),
        ),
        _ => (
            Cow::Owned(image.to_rgb8().into_raw()),
            3,
            b"DeviceRGB".as_slice(),
        ),
    }
}

fn encode_jpeg_stream(
    image: &DynamicImage,
    width: u32,
    height: u32,
    quality: u8,
) -> Result<EncodedPdfStream, PdfError> {
    let (bytes, color_space_name) = match image {
        DynamicImage::ImageLuma8(gray) => (
            encode_jpeg_gray(gray.as_raw(), width, height, quality)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG grayscale encode: {e}")))?,
            b"DeviceGray".as_slice(),
        ),
        DynamicImage::ImageRgb8(rgb) => (
            encode_jpeg_rgb(rgb.as_raw(), width, height, quality)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG RGB encode: {e}")))?,
            b"DeviceRGB".as_slice(),
        ),
        _ => {
            let rgb = image.to_rgb8();
            (
                encode_jpeg_rgb(rgb.as_raw(), width, height, quality)
                    .map_err(|e| PdfError::ImageEncode(format!("JPEG RGB encode: {e}")))?,
                b"DeviceRGB".as_slice(),
            )
        }
    };

    Ok(EncodedPdfStream {
        bytes,
        filter_name: b"DCTDecode",
        color_space_name,
        decode_parms: None,
        bits_per_component: 8,
    })
}

/// Encodes pixels as a native ISO 15444-1 JPEG 2000 codestream for PDF `/JPXDecode`.
fn encode_jpx_stream(
    image: &DynamicImage,
    width: u32,
    height: u32,
    quality: u8,
) -> Result<EncodedPdfStream, PdfError> {
    let (bytes, color_space_name) = match image {
        DynamicImage::ImageLuma8(gray) => (
            encode_jpx_gray(gray.as_raw(), width, height, quality)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG 2000 grayscale encode: {e}")))?,
            b"DeviceGray".as_slice(),
        ),
        DynamicImage::ImageRgb8(rgb) => (
            encode_jpx_rgb(rgb.as_raw(), width, height, quality)
                .map_err(|e| PdfError::ImageEncode(format!("JPEG 2000 RGB encode: {e}")))?,
            b"DeviceRGB".as_slice(),
        ),
        _ => {
            let rgb = image.to_rgb8();
            (
                encode_jpx_rgb(rgb.as_raw(), width, height, quality)
                    .map_err(|e| PdfError::ImageEncode(format!("JPEG 2000 RGB encode: {e}")))?,
                b"DeviceRGB".as_slice(),
            )
        }
    };

    Ok(EncodedPdfStream {
        bytes,
        filter_name: b"JPXDecode",
        color_space_name,
        decode_parms: None,
        bits_per_component: 8,
    })
}

/// Encodes lossless pixels as a valid PDF `/FlateDecode` stream.
/// Uses PNG optimal row filtering (`PngEncoder` IDAT payload -> PDF `/Predictor 15`) when available,
/// falling back to raw zlib compression if PNG chunk inspection does not match 8-bit L/RGB.
fn encode_pdf_flate_lossless(
    image: &DynamicImage,
    width: u32,
    height: u32,
) -> Result<EncodedPdfStream, PdfError> {
    let (raw_bytes, colors, color_space_name) = image_raw_luma_or_rgb(image);
    let ext_color = if colors == 1 {
        ExtendedColorType::L8
    } else {
        ExtendedColorType::Rgb8
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
            bits_per_component: 8,
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
        bits_per_component: 8,
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
    keep_smask: bool,
    keep_mask: bool,
) {
    stream.dict.set("Type", Object::Name(b"XObject".to_vec()));
    stream.dict.set("Subtype", Object::Name(b"Image".to_vec()));
    stream.dict.set("Width", Object::Integer(new_w as i64));
    stream.dict.set("Height", Object::Integer(new_h as i64));
    stream.dict.set(
        "BitsPerComponent",
        Object::Integer(encoded.bits_per_component as i64),
    );
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
    if !keep_smask {
        stream.dict.remove(b"SMask");
        stream.dict.remove(b"Matte");
    }
    if !keep_mask {
        stream.dict.remove(b"Mask");
    }
    stream.dict.remove(b"ImageMask");
    stream.dict.remove(b"Decode");
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

    fn test_noise_bytes(len: usize, mut seed: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                seed = seed
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((seed >> 32) & 0xFF) as u8
            })
            .collect()
    }

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
        let pixels = test_noise_bytes(100 * 100 * 3, 0x1122_3344_5566_7788);
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
        let cmyk_pixels = test_noise_bytes(100 * 100 * 4, 0x2233_4455_6677_8899);
        let cmyk_pdf = build_test_pdf_with_image(100, 100, "DeviceCMYK", cmyk_pixels, false);
        let opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
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
        let gray_pixels = test_noise_bytes(100 * 100, 0x3344_5566_7788_99AA);
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

        // Test SMask preservation and proportional scaling (instead of opaque white flattening).
        let mut doc = Document::load_mem(&build_test_pdf_with_image(
            100,
            100,
            "DeviceRGB",
            test_noise_bytes(100 * 100 * 3, 0x4455_6677_8899_AABB),
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

        let half_opts = CompressionOptions {
            scale: 0.5,
            ..opts.clone()
        };
        let (out_smask, stats_smask) =
            compress_pdf(&with_smask, &half_opts, |_| {}).expect("smask compression succeeds");
        assert_eq!(stats_smask.images_recompressed, 1);
        let parsed_smask = Document::load_mem(&out_smask).expect("valid output");
        let mut saw_scaled_parent = false;
        for obj in parsed_smask.objects.values() {
            if let Object::Stream(s) = obj {
                if is_image_xobject(&parsed_smask, s)
                    && s.dict.get(b"ColorSpace").and_then(|v| v.as_name()).ok() == Some(b"DeviceRGB")
                {
                    assert_eq!(s.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(50));
                    assert_eq!(s.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(50));
                    let smask_ref = s
                        .dict
                        .get(b"SMask")
                        .and_then(|v| v.as_reference())
                        .expect("parent keeps /SMask reference after renumber");
                    let smask_out = parsed_smask
                        .objects
                        .get(&smask_ref)
                        .and_then(|o| o.as_stream().ok())
                        .expect("referenced /SMask stream exists");
                    assert_eq!(
                        smask_out.dict.get(b"Width").and_then(|v| v.as_i64()).ok(),
                        Some(50)
                    );
                    assert_eq!(
                        smask_out.dict.get(b"Height").and_then(|v| v.as_i64()).ok(),
                        Some(50)
                    );
                    saw_scaled_parent = true;
                }
            }
        }
        assert!(saw_scaled_parent);
    }

    #[test]
    fn downscales_imagemask_and_ccitt_and_jbig2_and_jpx_streams() {
        // 1. 1-bit Flate /ImageMask true at scale = 0.10
        let mut doc = Document::load_mem(&build_test_pdf_with_image(
            200,
            200,
            "DeviceGray",
            vec![0u8; 200 * 200],
            false,
        ))
        .expect("load base pdf");
        let row_bytes = 200usize.div_ceil(8);
        let mut zlib = ZlibEncoder::new(Vec::new(), Compression::best());
        zlib.write_all(&test_noise_bytes(row_bytes * 200, 0x5566_7788_99AA_BBCC))
            .unwrap();
        let compressed_mask = zlib.finish().unwrap();
        for obj in doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if is_image_xobject(&Document::new(), s) {
                    s.dict.set("ImageMask", true);
                    s.dict.remove(b"ColorSpace");
                    s.dict.set("BitsPerComponent", 1);
                    s.dict.set("Filter", "FlateDecode");
                    s.content = compressed_mask.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut mask_pdf = Vec::new();
        doc.save_to(&mut mask_pdf).expect("save mask pdf");
        let summary_mask = inspect_pdf(&mask_pdf).expect("inspect mask");
        assert_eq!(summary_mask.image_count, 1);

        assert_eq!(summary_mask.compressible_image_count, 0);
        assert_eq!(summary_mask.skipped_1bit_count, 1);
        assert_eq!(summary_mask.fixed_image_bytes, compressed_mask.len());

        let low_scale_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.10,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let (out_mask, stats_mask) =
            compress_pdf(&mask_pdf, &low_scale_opts, |_| {}).expect("compress mask");
        assert_eq!(stats_mask.images_recompressed, 0);
        assert_eq!(stats_mask.images_skipped_small_or_mask, 1);
        assert_eq!(stats_mask.images_skipped_unsupported, 0);
        let parsed_mask = Document::load_mem(&out_mask).expect("parse mask output");
        let mask_stream = parsed_mask
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s)
                    if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") =>
                {
                    Some(s)
                }
                _ => None,
            })
            .expect("mask stream present");
        assert_eq!(mask_stream.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(200));
        assert_eq!(mask_stream.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(200));
        assert_eq!(mask_stream.dict.get(b"BitsPerComponent").and_then(|v| v.as_i64()).ok(), Some(1));
        assert_eq!(mask_stream.dict.get(b"ImageMask").and_then(|v| v.as_bool()).ok(), Some(true));
        assert_eq!(mask_stream.content, compressed_mask);

        // 2. /CCITTFaxDecode (K = -1 Group 4, all-white 100x4000 image -> 4000 V(0) '1' bits = 500 bytes)
        // Verify decoder accuracy for BlackIs1/Decode AND verify 1-bit CCITT streams stay untouched.
        let ccitt_g4 = vec![0xFFu8; 500];
        for &(black_is_1, use_decode_invert) in &[(false, false), (true, true)] {
            let mut ccitt_doc = Document::load_mem(&build_test_pdf_with_image(
                100,
                4000,
                "DeviceGray",
                vec![0u8; 100 * 4000],
                false,
            ))
            .expect("load base ccitt");
            for obj in ccitt_doc.objects.values_mut() {
                if let Object::Stream(s) = obj {
                    if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                        s.dict.set("BitsPerComponent", 1);
                        s.dict.set("Filter", "CCITTFaxDecode");
                        s.dict.set(
                            "DecodeParms",
                            dictionary! {
                                "K" => -1,
                                "Columns" => 100,
                                "Rows" => 4000,
                                "BlackIs1" => black_is_1,
                            },
                        );
                        if use_decode_invert {
                            s.dict.set("Decode", vec![1.into(), 0.into()]);
                        } else {
                            s.dict.remove(b"Decode");
                        }
                        s.content = ccitt_g4.clone();
                        s.allows_compression = false;
                    }
                }
            }
            let mut ccitt_pdf = Vec::new();
            ccitt_doc.save_to(&mut ccitt_pdf).expect("save ccitt");
            let summary = inspect_pdf(&ccitt_pdf).expect("inspect ccitt");
            assert_eq!(summary.compressible_image_count, 0);
            assert_eq!(summary.skipped_1bit_count, 1);
            assert_eq!(summary.fixed_image_bytes, ccitt_g4.len());

            let (out_ccitt, stats_ccitt) =
                compress_pdf(&ccitt_pdf, &low_scale_opts, |_| {}).expect("compress ccitt");
            assert_eq!(stats_ccitt.images_recompressed, 0);
            assert_eq!(stats_ccitt.images_skipped_small_or_mask, 1);
            assert_eq!(stats_ccitt.images_skipped_unsupported, 0);
            let parsed = Document::load_mem(&out_ccitt).expect("parse ccitt out");
            let img_s = parsed
                .objects
                .values()
                .find_map(|o| match o {
                    Object::Stream(s)
                        if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok()
                            == Some(b"Image") =>
                    {
                        Some(s)
                    }
                    _ => None,
                })
                .expect("output image");
            assert_eq!(img_s.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(100));
            assert_eq!(img_s.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(4000));
            assert_eq!(
                img_s.dict.get(b"Filter").and_then(|v| v.as_name()).ok(),
                Some(b"CCITTFaxDecode".as_slice())
            );
            assert_eq!(img_s.content, ccitt_g4);
            let decoded_out =
                decode_stream_to_image(&parsed, img_s, 100, 4000, false, false, None)
                    .expect("decode ccitt")
                    .into_luma8();
            assert!(
                decoded_out.as_raw().iter().all(|&p| p > 240),
                "expected white pixels after CCITT BlackIs1={black_is_1}, decode={use_decode_invert}"
            );
        }

        // Note: reuse small 13-byte all-white CCITT body for the 100x100 JBIG2 MMR test fixture below.
        let ccitt_g4_100 = [vec![0xFFu8; 12], vec![0xF0u8]].concat();
        // 3. /JBIG2Decode embedded page (PageInformation + ImmediateLosslessGenericRegion MMR=1)
        let mut jbig2_stream = Vec::new();
        // Segment 0: PageInformation (type 48), 19-byte data
        jbig2_stream.extend_from_slice(&0u32.to_be_bytes()); // segment number 0
        jbig2_stream.push(48); // flags: page info, 1-byte page assoc, short ref count 0
        jbig2_stream.push(0x00); // ref count = 0
        jbig2_stream.push(1); // page association = 1
        jbig2_stream.extend_from_slice(&19u32.to_be_bytes()); // data length = 19
        jbig2_stream.extend_from_slice(&100u32.to_be_bytes()); // width = 100
        jbig2_stream.extend_from_slice(&100u32.to_be_bytes()); // height = 100
        jbig2_stream.extend_from_slice(&0u32.to_be_bytes()); // x res
        jbig2_stream.extend_from_slice(&0u32.to_be_bytes()); // y res
        jbig2_stream.push(0x00); // page flags: default pixel 0, OR combination
        jbig2_stream.extend_from_slice(&0u16.to_be_bytes()); // striping 0
        // Segment 1: ImmediateLosslessGenericRegion (type 39), MMR = 1
        let region_header_len = 18u32; // 17 bytes region info + 1 byte MMR flags
        let data_len = region_header_len + ccitt_g4_100.len() as u32;
        jbig2_stream.extend_from_slice(&1u32.to_be_bytes()); // segment number 1
        jbig2_stream.push(39); // ImmediateLosslessGenericRegion
        jbig2_stream.push(0x00); // ref count = 0
        jbig2_stream.push(1); // page association = 1
        jbig2_stream.extend_from_slice(&data_len.to_be_bytes());
        jbig2_stream.extend_from_slice(&100u32.to_be_bytes()); // region width
        jbig2_stream.extend_from_slice(&100u32.to_be_bytes()); // region height
        jbig2_stream.extend_from_slice(&0u32.to_be_bytes()); // x
        jbig2_stream.extend_from_slice(&0u32.to_be_bytes()); // y
        jbig2_stream.push(0x00); // ext comb op = OR
        jbig2_stream.push(0x01); // generic region flags: MMR = 1
        jbig2_stream.extend_from_slice(&ccitt_g4_100);

        let mut jbig2_doc = Document::load_mem(&build_test_pdf_with_image(
            100,
            100,
            "DeviceGray",
            vec![0u8; 100 * 100],
            false,
        ))
        .expect("load base jbig2");
        for obj in jbig2_doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                    s.dict.set("BitsPerComponent", 1);
                    s.dict.set("Filter", "JBIG2Decode");
                    s.content = jbig2_stream.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut jbig2_pdf = Vec::new();
        jbig2_doc.save_to(&mut jbig2_pdf).expect("save jbig2");
        let jbig2_summary = inspect_pdf(&jbig2_pdf).expect("inspect jbig2");
        assert_eq!(jbig2_summary.compressible_image_count, 0);
        assert_eq!(jbig2_summary.skipped_1bit_count, 1);
        assert_eq!(jbig2_summary.fixed_image_bytes, jbig2_stream.len());
        let (out_jbig2, stats_jbig2) =
            compress_pdf(&jbig2_pdf, &low_scale_opts, |_| {}).expect("compress jbig2");
        assert_eq!(stats_jbig2.images_recompressed, 0);
        assert_eq!(stats_jbig2.images_skipped_small_or_mask, 1);
        assert_eq!(stats_jbig2.images_skipped_unsupported, 0);
        let parsed_jbig2 = Document::load_mem(&out_jbig2).expect("parse jbig2 out");
        let jbig2_out_stream = parsed_jbig2
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s)
                    if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") =>
                {
                    Some(s)
                }
                _ => None,
            })
            .expect("jbig2 out stream");
        assert_eq!(jbig2_out_stream.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(100));
        assert_eq!(jbig2_out_stream.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(100));
        assert_eq!(
            jbig2_out_stream.dict.get(b"Filter").and_then(|v| v.as_name()).ok(),
            Some(b"JBIG2Decode".as_slice())
        );
        assert_eq!(jbig2_out_stream.content, jbig2_stream);

        // 4. /JPXDecode with omitted /ColorSpace (single-component 100x100 minimal J2C codestream)
        let mut j2c: Vec<u8> = Vec::new();
        j2c.extend_from_slice(&[0xFF, 0x4F]); // SOC
        // SIZ marker: Lsiz=41 (1 component)
        j2c.extend_from_slice(&[0xFF, 0x51]);
        j2c.extend_from_slice(&41u16.to_be_bytes());
        j2c.extend_from_slice(&0u16.to_be_bytes()); // Rsiz
        j2c.extend_from_slice(&100u32.to_be_bytes()); // Xsiz
        j2c.extend_from_slice(&100u32.to_be_bytes()); // Ysiz
        j2c.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
        j2c.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
        j2c.extend_from_slice(&100u32.to_be_bytes()); // XTsiz
        j2c.extend_from_slice(&100u32.to_be_bytes()); // YTsiz
        j2c.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
        j2c.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
        j2c.extend_from_slice(&1u16.to_be_bytes()); // Csiz = 1
        j2c.extend_from_slice(&[7, 1, 1]); // 8-bit unsigned, XRsiz=1, YRsiz=1
        // COD marker: Lcod=12
        j2c.extend_from_slice(&[0xFF, 0x52]);
        j2c.extend_from_slice(&12u16.to_be_bytes());
        j2c.push(0x00); // Scod
        j2c.push(0x00); // LRCP progression
        j2c.extend_from_slice(&1u16.to_be_bytes()); // 1 layer
        j2c.push(0x00); // MCT = 0
        j2c.push(0x00); // decomposition levels = 0 (1 resolution level)
        j2c.push(0x04); // cb width exp - 2 -> 64
        j2c.push(0x04); // cb height exp - 2 -> 64
        j2c.push(0x00); // cb style
        j2c.push(0x01); // 5-3 reversible wavelet
        // QCD marker: Lqcd=4 (1 subband LL0, reversible, 0 guard bits)
        j2c.extend_from_slice(&[0xFF, 0x5C]);
        j2c.extend_from_slice(&4u16.to_be_bytes());
        j2c.push(0x40); // 2 guard bits, no quantization
        j2c.push(8 << 3); // epsilon_b = 8
        let mut j2c_padded = j2c.clone();
        // COM marker (1.2 KB payload so 0.10x JPEG (~350 B) genuinely saves space on padded JPX)
        j2c_padded.extend_from_slice(&[0xFF, 0x64]);
        j2c_padded.extend_from_slice(&1204u16.to_be_bytes()); // Lcom = 2 + Rcom(2) + 1200
        j2c_padded.extend_from_slice(&0u16.to_be_bytes()); // Rcom = binary
        j2c_padded.extend(std::iter::repeat_n(0u8, 1200));
        // SOT marker: Lsot=10
        let mut j2c_tail = Vec::new();
        j2c_tail.extend_from_slice(&[0xFF, 0x90]);
        j2c_tail.extend_from_slice(&10u16.to_be_bytes());
        j2c_tail.extend_from_slice(&0u16.to_be_bytes()); // Isot = 0
        // Psot: SOT(12) + SOD(2) + packet(2) = 16
        j2c_tail.extend_from_slice(&16u32.to_be_bytes());
        j2c_tail.push(0); // TPsot
        j2c_tail.push(1); // TNsot
        j2c_tail.extend_from_slice(&[0xFF, 0x93]); // SOD
        j2c_tail.extend_from_slice(&[0x00, 0x00]); // 4 zero-contribution packet bits for 4 codeblocks -> empty
        j2c_tail.extend_from_slice(&[0xFF, 0xD9]); // EOC
        let mut j2c_compact = j2c.clone();
        j2c_compact.extend_from_slice(&j2c_tail);
        j2c_padded.extend_from_slice(&j2c_tail);
        let j2c = j2c_padded;

        let mut jpx_doc = Document::load_mem(&build_test_pdf_with_image(
            100,
            100,
            "DeviceGray",
            vec![0u8; 100 * 100],
            false,
        ))
        .expect("load base jpx");
        for obj in jpx_doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                    s.dict.remove(b"ColorSpace");
                    s.dict.remove(b"BitsPerComponent");
                    s.dict.set("Filter", "JPXDecode");
                    s.content = j2c.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut jpx_pdf = Vec::new();
        jpx_doc.save_to(&mut jpx_pdf).expect("save jpx");
        let jpx_summary = inspect_pdf(&jpx_pdf).expect("inspect jpx");
        assert_eq!(jpx_summary.compressible_image_count, 1);
        let (out_jpx, stats_jpx) =
            compress_pdf(&jpx_pdf, &low_scale_opts, |_| {}).expect("compress jpx");
        assert_eq!(stats_jpx.images_recompressed, 1);
        assert_eq!(stats_jpx.images_skipped_unsupported, 0);
        // Verify save_modern() writes PDF 1.5+ /ObjStm and /XRef structures.
        assert!(
            out_jpx.windows(5).any(|w| w == b"/XRef")
                && out_jpx.windows(7).any(|w| w == b"/ObjStm"),
            "expected save_modern() output to contain /XRef and /ObjStm"
        );
        let parsed_jpx = Document::load_mem(&out_jpx).expect("parse jpx out");
        let jpx_out_stream = parsed_jpx
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s)
                    if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") =>
                {
                    Some(s)
                }
                _ => None,
            })
            .expect("jpx out stream");
        assert_eq!(jpx_out_stream.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(10));
        assert_eq!(jpx_out_stream.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(10));

        // 5. Combined MRC pattern: /JPXDecode 8-bit parent + 1-bit /JBIG2Decode /SMask
        //    - inspect_pdf partitions the 1-bit /SMask into skipped_1bit_count / fixed_image_bytes.
        //    - At scale = 0.10: parent scales to 10x10 and preserves the 1-bit /JBIG2Decode /SMask untouched.
        //    - At scale = 1.00: already-smaller compact JPX fails per-stream guard and stays untouched.
        let mut mrc_doc = Document::load_mem(&jpx_pdf).expect("reload jpx pdf");
        let jbig2_smask_id = mrc_doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100,
                "Height" => 100,
                "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 1,
                "Filter" => "JBIG2Decode",
            },
            jbig2_stream.clone(),
        ));
        for obj in mrc_doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Filter").and_then(|v| v.as_name()).ok() == Some(b"JPXDecode") {
                    s.dict.set("SMask", jbig2_smask_id);
                }
            }
        }
        let mut mrc_pdf = Vec::new();
        mrc_doc.save_to(&mut mrc_pdf).expect("save mrc pdf");
        let mrc_summary = inspect_pdf(&mrc_pdf).expect("inspect mrc pdf");
        assert_eq!(mrc_summary.compressible_image_count, 1);
        assert_eq!(mrc_summary.skipped_1bit_count, 1);
        assert_eq!(mrc_summary.fixed_image_bytes, jbig2_stream.len());

        let (out_mrc_scaled, stats_mrc_scaled) =
            compress_pdf(&mrc_pdf, &low_scale_opts, |_| {}).expect("compress mrc 0.10x");
        assert_eq!(stats_mrc_scaled.images_recompressed, 1);
        assert_eq!(stats_mrc_scaled.images_skipped_unsupported, 0);
        let parsed_mrc = Document::load_mem(&out_mrc_scaled).expect("parse mrc out");
        let mrc_parent = parsed_mrc
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s)
                    if s.dict.get(b"SMask").is_ok()
                        && s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") =>
                {
                    Some(s)
                }
                _ => None,
            })
            .expect("mrc parent out");
        assert_eq!(mrc_parent.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(10));
        assert_eq!(mrc_parent.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(10));
        let mrc_smask_ref = mrc_parent
            .dict
            .get(b"SMask")
            .and_then(|v| v.as_reference())
            .expect("mrc SMask ref");
        let mrc_smask_out = parsed_mrc
            .objects
            .get(&mrc_smask_ref)
            .and_then(|o| o.as_stream().ok())
            .expect("mrc SMask stream");
        assert_eq!(mrc_smask_out.dict.get(b"Width").and_then(|v| v.as_i64()).ok(), Some(100));
        assert_eq!(mrc_smask_out.dict.get(b"Height").and_then(|v| v.as_i64()).ok(), Some(100));
        assert_eq!(
            mrc_smask_out
                .dict
                .get(b"BitsPerComponent")
                .and_then(|v| v.as_i64())
                .ok(),
            Some(1)
        );
        assert_eq!(
            mrc_smask_out
                .dict
                .get(b"Filter")
                .and_then(|v| v.as_name())
                .ok(),
            Some(b"JBIG2Decode".as_slice())
        );
        assert_eq!(mrc_smask_out.content, jbig2_stream);

        // Verify scale = 1.00 keeps hyper-compact JPX (~90 B) + JBIG2 (~72 B) untouched instead of inflating.
        for obj in mrc_doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Filter").and_then(|v| v.as_name()).ok() == Some(b"JPXDecode") {
                    s.content = j2c_compact.clone();
                }
            }
        }
        let mut compact_mrc_pdf = Vec::new();
        mrc_doc.save_to(&mut compact_mrc_pdf).expect("save compact mrc");
        let full_scale_opts = CompressionOptions {
            scale: 1.0,
            ..low_scale_opts
        };
        let (_out_mrc_full, stats_mrc_full) =
            compress_pdf(&compact_mrc_pdf, &full_scale_opts, |_| {}).expect("compress mrc 1.00x");
        assert_eq!(stats_mrc_full.images_kept_original, 1);
        assert_eq!(stats_mrc_full.images_recompressed, 0);

        // Also verify: if compact parent fails per-stream bloat guard, its 8-bit /SMask is NOT mutated.
        let mut guard_mrc_doc = Document::load_mem(&compact_mrc_pdf).expect("reload compact mrc");
        let smask_noise = test_noise_bytes(100 * 100, 0x1122_3344_5566_7788);
        let mut smask_zlib = ZlibEncoder::new(Vec::new(), Compression::best());
        smask_zlib.write_all(&smask_noise).unwrap();
        let smask_flate_bytes = smask_zlib.finish().unwrap();
        let mut raw_8bit_smask = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100,
                "Height" => 100,
                "ColorSpace" => "DeviceGray",
                "BitsPerComponent" => 8,
                "Filter" => "FlateDecode",
            },
            smask_flate_bytes.clone(),
        );
        raw_8bit_smask.allows_compression = false;
        let raw_smask_id = guard_mrc_doc.add_object(raw_8bit_smask);
        for obj in guard_mrc_doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Filter").and_then(|v| v.as_name()).ok() == Some(b"JPXDecode") {
                    s.dict.set("SMask", raw_smask_id);
                }
            }
        }
        let mut guard_mrc_pdf = Vec::new();
        guard_mrc_doc.save_to(&mut guard_mrc_pdf).expect("save guard mrc");
        let (out_guard_mrc, stats_guard_mrc) =
            compress_pdf(&guard_mrc_pdf, &low_scale_opts, |_| {}).expect("compress guard mrc");
        assert_eq!(stats_guard_mrc.images_kept_original, 1);
        assert_eq!(stats_guard_mrc.images_recompressed, 0);
        let parsed_guard = Document::load_mem(&out_guard_mrc).expect("parse guard mrc");
        let guard_parent = parsed_guard
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s) if s.dict.get(b"SMask").is_ok() => Some(s),
                _ => None,
            })
            .expect("guard parent");
        let g_smask_ref = guard_parent.dict.get(b"SMask").unwrap().as_reference().unwrap();
        let g_smask = parsed_guard.objects.get(&g_smask_ref).unwrap().as_stream().unwrap();
        assert_eq!(g_smask.dict.get(b"Width").unwrap().as_i64().unwrap(), 100);
        assert_eq!(g_smask.dict.get(b"Height").unwrap().as_i64().unwrap(), 100);
        assert_eq!(g_smask.content, smask_flate_bytes);
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

        // PDFs with no compressible images still reflect structural + Flate + GC/dedup savings.
        let tiny_only = build_test_pdf_with_image(20, 20, "DeviceRGB", vec![128; 20 * 20 * 3], false);
        let tiny_summary = inspect_pdf(&tiny_only).expect("inspect tiny");
        assert_eq!(tiny_summary.compressible_image_count, 0);
        let tiny_est = tiny_summary.estimate_compressed_bytes(&opts);
        assert!(tiny_est > 0 && tiny_est <= tiny_only.len());
    }

    #[test]
    fn inverse_coupling_solvers_adjust_scale_and_quality_cleanly() {
        let synthetic_summary = PdfSummary::synthetic_for_test(
            10,
            10 * 1024 * 1024,
            200 * 1024,
            200,
            400,
            48_000,
            false,
        );
        let target_3mb = 3 * 1024 * 1024;

        // 1. Increasing quality should decrease solved scale to preserve target.
        let low_q_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 25,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let high_q_opts = CompressionOptions {
            quality: 85,
            ..low_q_opts.clone()
        };
        let scale_at_low_q = synthetic_summary.solve_scale_for_target(&low_q_opts, target_3mb);
        let scale_at_high_q = synthetic_summary.solve_scale_for_target(&high_q_opts, target_3mb);
        assert!(
            scale_at_low_q > scale_at_high_q,
            "expected scale_at_low_q ({scale_at_low_q}) > scale_at_high_q ({scale_at_high_q})"
        );

        // 2. Increasing scale should decrease solved quality to preserve target.
        let small_scale_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 0.30,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let large_scale_opts = CompressionOptions {
            scale: 0.65,
            ..small_scale_opts.clone()
        };
        let q_at_small_scale =
            synthetic_summary.solve_quality_for_target(&small_scale_opts, target_3mb);
        let q_at_large_scale =
            synthetic_summary.solve_quality_for_target(&large_scale_opts, target_3mb);
        assert!(
            q_at_small_scale > q_at_large_scale,
            "expected q_at_small_scale ({q_at_small_scale}) > q_at_large_scale ({q_at_large_scale})"
        );

        // 3. Target-first adjusts scale first, and only changes quality if scale hits bounds.
        let baseline_q65 = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 65,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let mut opts_mid = baseline_q65.clone();
        synthetic_summary.apply_target_scale_first(&mut opts_mid, target_3mb);
        assert_eq!(opts_mid.quality, 65);
        assert!(
            opts_mid.scale > 0.10 && opts_mid.scale < 1.00,
            "expected mid scale in (0.10, 1.00), got {}",
            opts_mid.scale
        );
        // Solved estimate tracks the 98.5% safety budget ceiling cleanly (`~90%-100%` of target_3mb).
        let solved_est = synthetic_summary.estimate_compressed_bytes(&opts_mid);
        assert!(
            solved_est >= (target_3mb * 85) / 100 && solved_est <= target_3mb,
            "solved_est {solved_est} should utilize target budget {target_3mb} without overshoot"
        );

        // Target below 0.10x @ Q65: hits 0.10x lower bound and lowers quality below 65.
        let mut opts_tiny_target = baseline_q65.clone();
        synthetic_summary.apply_target_scale_first(&mut opts_tiny_target, 300 * 1024);
        assert!((opts_tiny_target.scale - 0.10).abs() < 1e-4);
        assert!(opts_tiny_target.quality < 65);

        // Target above 1.00x @ Q65: hits 1.00x upper bound and raises quality above 65.
        let mut opts_large_target = baseline_q65;
        synthetic_summary.apply_target_scale_first(&mut opts_large_target, 15 * 1024 * 1024);
        assert!((opts_large_target.scale - 1.00).abs() < 1e-4);
        assert!(opts_large_target.quality > 65);

        // Default options start at Q100 / 1.00x / JPEG / Lanczos3.
        let defs = CompressionOptions::default();
        assert_eq!(defs.quality, 100);
        assert!((defs.scale - 1.00).abs() < 1e-6);
        assert_eq!(defs.format, ImageFormatChoice::Jpeg);
        assert_eq!(defs.resize_filter, ResizeFilterChoice::Lanczos3);

        // 4. Lossless PNG leaves quality untouched and only tunes scale.
        let mut opts_png = CompressionOptions {
            format: ImageFormatChoice::Png,
            quality: 42,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        synthetic_summary.apply_target_scale_first(&mut opts_png, target_3mb);
        assert_eq!(opts_png.quality, 42);
        assert!(opts_png.scale > 0.10 && opts_png.scale < 1.00);
    }

    #[test]
    fn deduplicates_shared_xobjects_across_many_pages_in_estimator_and_processor() {
        let mut doc = Document::load_mem(&build_test_pdf_with_image(
            200,
            200,
            "DeviceGray",
            vec![180u8; 200 * 200],
            false,
        ))
        .expect("load base pdf");

        // Find the existing page, resources, content, and image object IDs.
        let orig_pages = doc.get_pages();
        let page1_id = orig_pages[&1];
        let pages_id = doc
            .get_dictionary(page1_id)
            .expect("page1")
            .get(b"Parent")
            .expect("parent")
            .as_reference()
            .expect("parent ref");
        let p1_clone = doc.get_dictionary(page1_id).expect("page1 dict").clone();
        let res_obj = p1_clone.get(b"Resources").expect("res").clone();
        let cont_obj = p1_clone.get(b"Contents").expect("cont").clone();

        // Also wrap the shared image inside a Form XObject referenced alongside page resources.
        let img_id = doc
            .objects
            .iter()
            .find_map(|(&id, obj)| match obj {
                Object::Stream(s) if is_image_xobject(&doc, s) => Some(id),
                _ => None,
            })
            .expect("image id");
        let form_stream = Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Form",
                "BBox" => vec![0.into(), 0.into(), 100.into(), 100.into()],
                "Resources" => dictionary! {
                    "XObject" => dictionary! { "SharedImg" => img_id }
                },
            },
            b"q 100 0 0 100 0 0 cm /SharedImg Do Q".to_vec(),
        );
        let form_id = doc.add_object(form_stream);

        let mut kids = vec![Object::Reference(page1_id)];
        for idx in 1..50 {
            let p_res_id = doc.add_object(dictionary! {
                "XObject" => dictionary! {
                    "Im1" => img_id,
                    "FormOverlay" => form_id,
                }
            });
            let mut new_page = p1_clone.clone();
            if idx % 2 == 0 {
                new_page.set("Resources", p_res_id);
            } else {
                new_page.set("Resources", res_obj.clone());
            }
            new_page.set("Contents", cont_obj.clone());
            let pid = doc.add_object(new_page);
            kids.push(Object::Reference(pid));
        }
        if let Ok(Object::Dictionary(pages_dict)) = doc.get_object_mut(pages_id) {
            pages_dict.set("Count", 50);
            pages_dict.set("Kids", Object::Array(kids));
        }

        let mut shared_pdf = Vec::new();
        doc.save_to(&mut shared_pdf).expect("save 50-page shared pdf");

        let summary = inspect_pdf(&shared_pdf).expect("inspect shared");
        assert_eq!(summary.page_count, 50);
        assert_eq!(summary.image_count, 1);
        assert_eq!(summary.compressible_image_count, 1);
        assert_eq!(summary.compressible_gray_pixels, 200 * 200);

        let opts = CompressionOptions::default();
        let (out, stats) = compress_pdf(&shared_pdf, &opts, |_| {}).expect("compress shared");
        assert_eq!(stats.total_image_objects, 1);
        assert_eq!(stats.images_recompressed + stats.images_kept_original, 1);
        assert!(out.len() < shared_pdf.len() + 4096);
    }

    #[test]
    fn preserves_grayscale_and_1bit_channels_across_all_formats() {
        // Use noisy grayscale pixels so compressed output exceeds 1 KB slack & is clearly compressed.
        let gray_pixels = test_noise_bytes(200 * 200, 0x9E37_79B9_7F4A_7C15);
        let gray_pdf = build_test_pdf_with_image(200, 200, "DeviceGray", gray_pixels, false);
        let summary = inspect_pdf(&gray_pdf).expect("inspect gray");
        assert_eq!(summary.compressible_gray_pixels, 200 * 200);
        assert_eq!(summary.compressible_rgb_pixels, 0);

        for fmt in ImageFormatChoice::ALL {
            let opts = CompressionOptions {
                format: fmt,
                quality: 50,
                scale: 0.5,
                resize_filter: ResizeFilterChoice::Lanczos3,
            };
            let (out, stats) = compress_pdf(&gray_pdf, &opts, |_| {}).expect("compress gray");
            assert_eq!(stats.images_recompressed, 1, "fmt={:?} stats={:?}", fmt, stats);
            let parsed = Document::load_mem(&out).expect("parse gray out");
            for obj in parsed.objects.values() {
                if let Object::Stream(s) = obj {
                    if is_image_xobject(&parsed, s) {
                        assert_eq!(
                            s.dict.get(b"ColorSpace").and_then(|v| v.as_name()).ok(),
                            Some(b"DeviceGray".as_slice()),
                            "format {:?} must keep DeviceGray, not RGB",
                            fmt
                        );
                    }
                }
            }
        }

        // Now verify 1-bit monochrome stream is skipped completely untouched (never expands to 8/24-bit).
        let mut doc = Document::load_mem(&gray_pdf).expect("load base");
        let row_bytes = 200usize.div_ceil(8);
        let packed: Vec<u8> = (0..(row_bytes * 200))
            .map(|i| (((i * 73) ^ (i >> 3)) & 0xFF) as u8)
            .collect();
        let mut zlib = ZlibEncoder::new(Vec::new(), Compression::best());
        zlib.write_all(&packed).unwrap();
        let packed_flate = zlib.finish().unwrap();
        for obj in doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                    s.dict.set("BitsPerComponent", 1);
                    s.dict.set("ColorSpace", "DeviceGray");
                    s.dict.set("Filter", "FlateDecode");
                    s.content = packed_flate.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut mono_pdf = Vec::new();
        doc.save_to(&mut mono_pdf).expect("save mono");
        let mono_summary = inspect_pdf(&mono_pdf).expect("inspect mono");
        assert_eq!(mono_summary.skipped_1bit_count, 1);
        assert_eq!(mono_summary.compressible_image_count, 0);
        assert_eq!(mono_summary.fixed_image_bytes, packed_flate.len());

        for fmt in ImageFormatChoice::ALL {
            let opts = CompressionOptions {
                format: fmt,
                quality: 50,
                scale: 0.5,
                resize_filter: ResizeFilterChoice::Lanczos3,
            };
            let (out, stats) = compress_pdf(&mono_pdf, &opts, |_| {}).expect("compress mono");
            assert_eq!(stats.images_skipped_small_or_mask, 1);
            assert_eq!(stats.images_recompressed, 0);
            let parsed = Document::load_mem(&out).expect("parse mono out");
            for obj in parsed.objects.values() {
                if let Object::Stream(s) = obj {
                    if is_image_xobject(&parsed, s) {
                        assert_eq!(
                            s.dict.get(b"ColorSpace").and_then(|v| v.as_name()).ok(),
                            Some(b"DeviceGray".as_slice())
                        );
                        assert_eq!(
                            s.dict.get(b"BitsPerComponent").and_then(|v| v.as_i64()).ok(),
                            Some(1)
                        );
                        assert_eq!(s.content, packed_flate);
                    }
                }
            }
        }
    }

    #[test]
    fn never_inflates_compact_streams_or_estimates_across_scales_and_formats() {
        // Test 1: Hyper-compact 1-bit Flate stream (~4.5 KB) stays untouched with no bloat.
        let mut doc = Document::load_mem(&build_test_pdf_with_image(
            800,
            800,
            "DeviceGray",
            vec![0u8; 800 * 800],
            false,
        ))
        .expect("load base");
        let row_bytes = 800usize.div_ceil(8);
        let mut raw_1bit = vec![0xAAu8; row_bytes * 800];
        for y in 0..800 {
            raw_1bit[y * row_bytes + (y % row_bytes)] ^= ((y * 13) & 0xFF) as u8;
            raw_1bit[y * row_bytes + ((y * 3) % row_bytes)] ^= ((y * 29) & 0xFF) as u8;
        }
        let mut zlib = ZlibEncoder::new(Vec::new(), Compression::best());
        zlib.write_all(&raw_1bit).unwrap();
        let compressed_1bit = zlib.finish().unwrap();
        assert!(compressed_1bit.len() > 2000);

        for obj in doc.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                    s.dict.set("BitsPerComponent", 1);
                    s.dict.set("ColorSpace", "DeviceGray");
                    s.dict.set("Filter", "FlateDecode");
                    s.content = compressed_1bit.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut input = Vec::new();
        doc.save_to(&mut input).expect("save compact mono pdf");

        let summary = inspect_pdf(&input).expect("inspect compact 1bit");
        for fmt in ImageFormatChoice::ALL {
            for &scale in &[0.50f32, 0.80, 1.00] {
                let opts = CompressionOptions {
                    format: fmt,
                    quality: 90,
                    scale,
                    resize_filter: ResizeFilterChoice::Lanczos3,
                };
                let est = summary.estimate_compressed_bytes(&opts);
                assert!(
                    est <= summary.file_bytes,
                    "1-bit estimator bloated ({est} > {}) for {:?} scale={scale}",
                    summary.file_bytes,
                    fmt
                );

                let (out, _stats) = compress_pdf(&input, &opts, |_| {}).expect("compress");
                assert!(
                    out.len() <= input.len() + 2048,
                    "1-bit output bloated ({} > {}) for {:?} scale={scale}",
                    out.len(),
                    input.len(),
                    fmt
                );
            }
        }

        // Test 2: Compact 8-bit repeating grayscale stream (~2.5 KB FlateDecode for 800x800)
        // must also never bloat above its original size under strict per-stream & estimator guards.
        let mut raw_8bit = vec![240u8; 800 * 800];
        for y in 0..800 {
            for x in (0..800).step_by(32) {
                raw_8bit[y * 800 + x] = ((x + y) & 0xFF) as u8;
            }
        }
        let mut zlib8 = ZlibEncoder::new(Vec::new(), Compression::best());
        zlib8.write_all(&raw_8bit).unwrap();
        let compressed_8bit = zlib8.finish().unwrap();

        let mut doc8 = Document::load_mem(&build_test_pdf_with_image(
            800,
            800,
            "DeviceGray",
            vec![0u8; 800 * 800],
            false,
        ))
        .expect("load base 8bit");
        for obj in doc8.objects.values_mut() {
            if let Object::Stream(s) = obj {
                if s.dict.get(b"Subtype").and_then(|v| v.as_name()).ok() == Some(b"Image") {
                    s.dict.set("BitsPerComponent", 8);
                    s.dict.set("ColorSpace", "DeviceGray");
                    s.dict.set("Filter", "FlateDecode");
                    s.content = compressed_8bit.clone();
                    s.allows_compression = false;
                }
            }
        }
        let mut input8 = Vec::new();
        doc8.save_to(&mut input8).expect("save compact 8bit pdf");

        let summary8 = inspect_pdf(&input8).expect("inspect compact 8bit");
        assert_eq!(summary8.compressible_image_count, 1);
        for fmt in ImageFormatChoice::ALL {
            for &scale in &[0.50f32, 0.80, 1.00] {
                let opts = CompressionOptions {
                    format: fmt,
                    quality: 90,
                    scale,
                    resize_filter: ResizeFilterChoice::Lanczos3,
                };
                let est = summary8.estimate_compressed_bytes(&opts);
                assert!(
                    est <= summary8.file_bytes,
                    "8-bit estimator bloated ({est} > {}) for {:?} scale={scale}",
                    summary8.file_bytes,
                    fmt
                );

                let (out8, _stats8) = compress_pdf(&input8, &opts, |_| {}).expect("compress 8bit");
                assert!(
                    out8.len() <= input8.len() + 2048,
                    "8-bit output bloated ({} > {}) for {:?} scale={scale}",
                    out8.len(),
                    input8.len(),
                    fmt
                );
            }
        }
    }

    #[test]
    fn smart_passthrough_preserves_dct_at_q100_scale_1() {
        let mut rgb = vec![0u8; 128 * 128 * 3];
        for y in 0..128usize {
            for x in 0..128usize {
                let p = (y * 128 + x) * 3;
                rgb[p] = ((x * 2) % 256) as u8;
                rgb[p + 1] = ((y * 2) % 256) as u8;
                rgb[p + 2] = (((x + y) * 3) % 256) as u8;
            }
        }
        let jpeg_payload =
            encode_jpeg_rgb(&rgb, 128, 128, 95).expect("encode test jpeg");

        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let img_id = doc.add_object(
            Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => 128,
                    "Height" => 128,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                    "Filter" => "DCTDecode",
                    "Length" => jpeg_payload.len() as i64,
                },
                jpeg_payload.clone(),
            )
            .with_compression(false),
        );
        let res_id = doc.add_object(dictionary! {
            "XObject" => dictionary! { "Im1" => img_id },
        });
        let cont_id = doc.add_object(Stream::new(
            dictionary! {},
            b"q 128 0 0 128 0 0 cm /Im1 Do Q".to_vec(),
        ));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => res_id,
            "Contents" => cont_id,
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 1,
                "Kids" => vec![Object::Reference(page_id)],
            }),
        );
        let cat_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", cat_id);
        let mut dct_pdf = Vec::new();
        doc.save_to(&mut dct_pdf).expect("save dct test pdf");

        let default_opts = CompressionOptions::default();
        let (out_passthrough, stats_passthrough) =
            compress_pdf(&dct_pdf, &default_opts, |_| {}).expect("passthrough compress");
        assert_eq!(stats_passthrough.images_kept_original, 1);
        assert_eq!(stats_passthrough.images_recompressed, 0);

        let parsed = Document::load_mem(&out_passthrough).expect("parse passthrough");
        let out_stream = parsed
            .objects
            .values()
            .find_map(|o| match o {
                Object::Stream(s) if is_image_xobject(&parsed, s) => Some(s),
                _ => None,
            })
            .expect("image stream");
        assert_eq!(out_stream.content, jpeg_payload);

        let recompress_opts = CompressionOptions {
            quality: 50,
            ..default_opts
        };
        let (_out_q50, stats_q50) =
            compress_pdf(&dct_pdf, &recompress_opts, |_| {}).expect("q50 compress");
        assert_eq!(stats_q50.images_recompressed, 1);
    }

    #[test]
    fn handles_empty_corrupt_encrypted_and_zero_image_pdfs() {
        assert!(matches!(inspect_pdf(b""), Err(PdfError::Parse(_))));
        assert!(matches!(
            compress_pdf(b"", &CompressionOptions::default(), |_| {}),
            Err(PdfError::Parse(_))
        ));

        // Encrypted trailer PDF surfaces PdfError::Encrypted cleanly.
        let mut doc_enc = Document::with_version("1.5");
        let pages_id = doc_enc.new_object_id();
        let cat_id = doc_enc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        doc_enc
            .objects
            .insert(pages_id, Object::Dictionary(dictionary! { "Type" => "Pages", "Count" => 0, "Kids" => Vec::<Object>::new() }));
        doc_enc.trailer.set("Root", cat_id);
        let enc_dict_id = doc_enc.add_object(dictionary! { "Filter" => "Standard", "V" => 2, "R" => 3, "Length" => 128 });
        doc_enc.trailer.set("Encrypt", enc_dict_id);
        let mut enc_bytes = Vec::new();
        doc_enc.save_to(&mut enc_bytes).expect("save enc mock");
        assert!(matches!(inspect_pdf(&enc_bytes), Err(PdfError::Encrypted)));
        assert!(matches!(
            compress_pdf(&enc_bytes, &CompressionOptions::default(), |_| {}),
            Err(PdfError::Encrypted)
        ));

        // Valid 0-image PDF compresses cleanly via save_modern().
        let mut doc_zero = Document::with_version("1.4");
        let p_root = doc_zero.new_object_id();
        let content_id = doc_zero.add_object(Stream::new(dictionary! {}, b"BT /F1 12 Tf ET".to_vec()));
        let page_id = doc_zero.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => p_root,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => content_id,
        });
        doc_zero.objects.insert(
            p_root,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 1,
                "Kids" => vec![Object::Reference(page_id)],
            }),
        );
        let c_id = doc_zero.add_object(dictionary! { "Type" => "Catalog", "Pages" => p_root });
        doc_zero.trailer.set("Root", c_id);
        let mut zero_bytes = Vec::new();
        doc_zero.save_to(&mut zero_bytes).expect("save 0-img pdf");

        let zero_summary = inspect_pdf(&zero_bytes).expect("inspect 0-img");
        assert_eq!(zero_summary.image_count, 0);
        let nan_opts = CompressionOptions {
            scale: f32::NAN,
            ..CompressionOptions::default()
        };
        let zero_est = zero_summary.estimate_compressed_bytes(&nan_opts);
        assert!(zero_est > 0 && zero_est <= zero_bytes.len());

        let (zero_out, zero_stats) = compress_pdf(&zero_bytes, &nan_opts, |_| {}).expect("0-img ok");
        assert_eq!(zero_stats.total_image_objects, 0);
        assert_eq!(zero_stats.images_recompressed, 0);
        assert!(Document::load_mem(&zero_out).is_ok());
    }

    #[test]
    fn parallel_multi_image_determinism_and_stream_lengths_synced() {
        let mut doc = Document::with_version("1.5");
        let pages_id = doc.new_object_id();
        let mut xobj_dict = Dictionary::new();
        let mut content_ops = Vec::new();
        for idx in 0..6u8 {
            let w = 120u32;
            let h = 120u32;
            let mut px = vec![0u8; (w * h * 3) as usize];
            for y in 0..h {
                for x in 0..w {
                    let p = ((y * w + x) * 3) as usize;
                    px[p] = (((x + (idx as u32) * 20) * 3) % 256) as u8;
                    px[p + 1] = (((y + (idx as u32) * 15) * 4) % 256) as u8;
                    px[p + 2] = ((x ^ y) % 256) as u8;
                }
            }
            let img_id = doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => w as i64,
                    "Height" => h as i64,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                },
                px,
            ));
            let name = format!("Im{idx}");
            xobj_dict.set(name.clone(), Object::Reference(img_id));
            content_ops.extend_from_slice(format!("q 100 0 0 100 0 0 cm /{name} Do Q\n").as_bytes());
        }
        let res_id = doc.add_object(dictionary! { "XObject" => xobj_dict });
        let cont_id = doc.add_object(Stream::new(dictionary! {}, content_ops));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => pages_id,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Resources" => res_id,
            "Contents" => cont_id,
        });
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 1,
                "Kids" => vec![Object::Reference(page_id)],
            }),
        );
        let cat_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", cat_id);
        let mut pdf_bytes = Vec::new();
        doc.save_to(&mut pdf_bytes).expect("save 6-img pdf");

        let opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 75,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let (out, stats) = compress_pdf(&pdf_bytes, &opts, |_| {}).expect("parallel compress");
        assert_eq!(stats.total_image_objects, 6);
        assert_eq!(stats.images_recompressed, 6);

        let parsed = Document::load_mem(&out).expect("parse compressed output");
        for obj in parsed.objects.values() {
            if let Object::Stream(s) = obj {
                let decl_len = s
                    .dict
                    .get(b"Length")
                    .and_then(|o| o.as_i64())
                    .expect("every stream must have integer /Length");
                assert_eq!(decl_len as usize, s.content.len());
            }
        }
    }

    #[test]
    fn three_stage_structural_gc_dedup_objstm_and_flate_pipeline() {
        let mut doc = Document::with_version("1.4");
        let pages_id = doc.new_object_id();

        // Identical font/CMap/vector payload compressed at two different zlib levels (Fast vs Best)
        // plus an uncompressed bloated content/font payload for Priority 1 + Priority 3 testing.
        let mut repeated_payload = Vec::with_capacity(16_384);
        for i in 0..256 {
            repeated_payload.extend_from_slice(
                format!("BT /F1 12 Tf 72 {} Td (Hello structural pipeline row {i}) Tj ET\n", 750 - (i % 60) * 10)
                    .as_bytes(),
            );
        }

        let mut fast_enc = ZlibEncoder::new(Vec::new(), Compression::fast());
        fast_enc.write_all(&repeated_payload).unwrap();
        let fast_flate = fast_enc.finish().unwrap();

        let mut best_enc = ZlibEncoder::new(Vec::new(), Compression::best());
        best_enc.write_all(&repeated_payload).unwrap();
        let best_flate = best_enc.finish().unwrap();
        assert!(best_flate.len() <= fast_flate.len());

        let icc_raw: Vec<u8> = (0..4096).map(|i| ((i * 17) % 251) as u8).collect();
        let mut icc_fast = ZlibEncoder::new(Vec::new(), Compression::fast());
        icc_fast.write_all(&icc_raw).unwrap();
        let icc_fast_bytes = icc_fast.finish().unwrap();

        let mut icc_none_flate = Stream::new(
            dictionary! { "N" => 3, "Filter" => "FlateDecode" },
            icc_fast_bytes.clone(),
        );
        icc_none_flate.allows_compression = false;

        // Duplicated indirect ICC streams, ColorSpace arrays, FontDescriptors, Font dicts, ExtGState.
        let icc1_id = doc.add_object(icc_none_flate);
        let icc2_id = doc.add_object(Stream::new(dictionary! { "N" => 3 }, icc_raw.clone()));
        let cs1_id = doc.add_object(vec![Object::Name(b"ICCBased".to_vec()), icc1_id.into()]);
        let cs2_id = doc.add_object(vec![Object::Name(b"ICCBased".to_vec()), icc2_id.into()]);

        let mut ff1 = Stream::new(
            dictionary! { "Filter" => "FlateDecode", "Subtype" => "Type1C" },
            fast_flate.clone(),
        );
        ff1.allows_compression = false;
        let mut ff2 = Stream::new(
            dictionary! { "Filter" => "FlateDecode", "Subtype" => "Type1C" },
            best_flate.clone(),
        );
        ff2.allows_compression = false;
        let ff1_id = doc.add_object(ff1);
        let ff2_id = doc.add_object(ff2);

        let fd1_id = doc.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => "TestSans",
            "FontFile3" => ff1_id,
        });
        let fd2_id = doc.add_object(dictionary! {
            "Type" => "FontDescriptor",
            "FontName" => "TestSans",
            "FontFile3" => ff2_id,
        });
        let font1_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "TestSans",
            "FontDescriptor" => fd1_id,
        });
        let font2_id = doc.add_object(dictionary! {
            "Type" => "Font",
            "Subtype" => "Type1",
            "BaseFont" => "TestSans",
            "FontDescriptor" => fd2_id,
        });
        // Non-zero generation indirect ExtGState (both identical!).
        let gs1_id: ObjectId = (900, 2);
        doc.objects.insert(
            gs1_id,
            Object::Dictionary(dictionary! { "Type" => "ExtGState", "CA" => 0.85f32 }),
        );
        let gs2_id = doc.add_object(dictionary! { "Type" => "ExtGState", "CA" => 0.85f32 });
        doc.max_id = doc.max_id.max(900);

        // Bloated Flate stream with /DecodeParms predictor preserved bit-for-bit across Priority 3.
        let mut pred_raw = Vec::with_capacity(4096);
        for row in 0..32usize {
            pred_raw.push(1); // PNG Sub filter byte
            for col in 0..64usize {
                pred_raw.push(((row * 3 + col) & 15) as u8);
            }
        }
        let mut pred_fast = ZlibEncoder::new(Vec::new(), Compression::fast());
        pred_fast.write_all(&pred_raw).unwrap();
        let pred_fast_bytes = pred_fast.finish().unwrap();
        let mut pred_stream = Stream::new(
            dictionary! {
                "Filter" => "FlateDecode",
                "DecodeParms" => dictionary! { "Predictor" => 11, "Columns" => 64, "Colors" => 1 },
            },
            pred_fast_bytes,
        );
        pred_stream.allows_compression = false;
        let pred_id = doc.add_object(pred_stream);

        // Duplicate identical 100x100 RGB image XObjects on different pages.
        let mut img_px = vec![0u8; 100 * 100 * 3];
        for y in 0..100usize {
            for x in 0..100usize {
                let p = (y * 100 + x) * 3;
                img_px[p] = ((x * 5) % 256) as u8;
                img_px[p + 1] = ((y * 7) % 256) as u8;
                img_px[p + 2] = (((x + y) * 3) % 256) as u8;
            }
        }
        let img1_id = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100,
                "Height" => 100,
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
            },
            img_px.clone(),
        ));
        let img2_id = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 100,
                "Height" => 100,
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
            },
            img_px,
        ));

        // 3 separate identical-ish pages (/Type /Page must remain 3 distinct objects!).
        let mut page_ids = Vec::new();
        for idx in 0..3usize {
            let (font_id, cs_id, gs_id, img_id) = if idx == 0 {
                (font1_id, cs1_id, gs1_id, img1_id)
            } else {
                (font2_id, cs2_id, gs2_id, img2_id)
            };
            let mut c_dict = dictionary! {
                "Filter" => "FlateDecode",
                "ExtraRef" => pred_id,
            };
            let _ = &mut c_dict;
            let content_id = doc.add_object(Stream::new(
                dictionary! {},
                b"q 100 0 0 100 0 0 cm /Im Do Q\nBT /F1 12 Tf 50 700 Td (Page) Tj ET".to_vec(),
            ));
            let res_id = doc.add_object(dictionary! {
                "Font" => dictionary! { "F1" => font_id },
                "ColorSpace" => dictionary! { "CS1" => cs_id },
                "ExtGState" => dictionary! { "GS1" => gs_id },
                "XObject" => dictionary! { "Im" => img_id, "RefPred" => pred_id },
            });
            let page_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Resources" => res_id,
                "Contents" => content_id,
            });
            page_ids.push(Object::Reference(page_id));
        }

        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 3,
                "Kids" => page_ids,
            }),
        );
        let cat_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        let info_id = doc.add_object(dictionary! { "Title" => Object::string_literal("Pipeline Test") });
        doc.trailer.set("Root", cat_id);
        doc.trailer.set("Info", info_id);

        // Orphaned objects (disconnected from /Root and /Info) representing stale incremental data.
        let _orphan_img = doc.add_object(Stream::new(
            dictionary! {
                "Type" => "XObject",
                "Subtype" => "Image",
                "Width" => 80,
                "Height" => 80,
                "ColorSpace" => "DeviceRGB",
                "BitsPerComponent" => 8,
            },
            vec![77u8; 80 * 80 * 3],
        ));
        let _orphan_dict = doc.add_object(dictionary! { "OldRev" => "stale orphan metadata" });

        let mut pdf_bytes = Vec::new();
        doc.save_to(&mut pdf_bytes).expect("save baseline v1.4 pdf");

        let summary = inspect_pdf(&pdf_bytes).expect("inspect succeeds");
        assert!(summary.gc_removed_objects >= 2, "expected orphans GC'd");
        assert!(summary.dedup_merged_objects >= 5, "expected cascading dedup merges");
        assert_eq!(summary.compressible_image_count, 1, "duplicate images counted once");
        assert!(summary.dedup_flate_stream_bytes > 0);
        assert!(summary.structural_dict_bytes > 0);

        let opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 80,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        assert!(summary.estimate_compressed_bytes(&opts) < pdf_bytes.len());

        let (out_bytes, stats) = compress_pdf(&pdf_bytes, &opts, |_| {}).expect("compress ok");
        assert!(stats.objects_pruned_gc >= 2);
        assert!(stats.objects_deduplicated >= 5);
        assert!(stats.flate_streams_recompressed >= 1);
        assert_eq!(stats.images_recompressed, 1, "only single deduplicated image processed");
        assert!(out_bytes.len() < pdf_bytes.len());

        // Priority 2 checks: header bumped >= 1.5, both /ObjStm and /XRef present & FlateDecode.
        let hdr_end = out_bytes.iter().position(|&b| b == b'\n').unwrap_or(16).min(16);
        let hdr = std::str::from_utf8(&out_bytes[..hdr_end]).unwrap_or("");
        assert!(hdr >= "%PDF-1.5", "expected PDF header >= 1.5, got {hdr:?}");
        assert!(out_bytes.windows(7).any(|w| w == b"/ObjStm"));
        assert!(out_bytes.windows(5).any(|w| w == b"/XRef"));

        // Verify /XRef itself carries /Filter/FlateDecode when smaller.
        let startxref_pos = find_subslice_rev(&out_bytes, b"startxref").expect("startxref");
        let after = std::str::from_utf8(&out_bytes[startxref_pos + 9..]).unwrap();
        let xref_off: usize = after.split("%%EOF").next().unwrap().trim().parse().unwrap();
        let xref_obj_hdr = &out_bytes[xref_off..startxref_pos];
        assert!(
            find_subslice(xref_obj_hdr, b"/Filter/FlateDecode").is_some(),
            "expected /Filter/FlateDecode on /XRef stream"
        );

        // Reload and verify structural validity, 3 distinct pages, /Info preserved, predictor lossless.
        let reloaded = Document::load_mem(&out_bytes).expect("reloaded valid PDF 1.5+");
        assert_eq!(reloaded.get_pages().len(), 3);
        assert!(reloaded.trailer.get(b"Info").is_ok());

        let reloaded_preds: Vec<&Stream> = reloaded
            .objects
            .values()
            .filter_map(|o| match o {
                Object::Stream(s) if s.dict.has(b"DecodeParms") => Some(s),
                _ => None,
            })
            .collect();
        assert_eq!(reloaded_preds.len(), 1);
        let mut dec = ZlibDecoder::new(reloaded_preds[0].content.as_slice());
        let mut roundtrip_pred = Vec::new();
        dec.read_to_end(&mut roundtrip_pred).expect("zlib decode pred");
        assert_eq!(roundtrip_pred, pred_raw);
    }

    #[test]
    fn estimator_tracks_structural_dedup_gc_and_flate_savings_without_images() {
        let mut doc = Document::with_version("1.4");
        let pages_id = doc.new_object_id();

        let font_data: Vec<u8> = (0..12_288).map(|i| ((i / 16) % 64 + 65) as u8).collect();
        let mut fast = ZlibEncoder::new(Vec::new(), Compression::fast());
        fast.write_all(&font_data).unwrap();
        let fast_bytes = fast.finish().unwrap();

        let mut kids = Vec::new();
        for _ in 0..8 {
            let mut s = Stream::new(dictionary! { "Filter" => "FlateDecode" }, fast_bytes.clone());
            s.allows_compression = false;
            let f_id = doc.add_object(s);
            let font_id = doc.add_object(dictionary! { "Type" => "Font", "FontFile2" => f_id });
            let c_id = doc.add_object(Stream::new(
                dictionary! {},
                b"BT /F1 11 Tf 72 720 Td (Repeated text block) Tj ET".repeat(20),
            ));
            let r_id = doc.add_object(dictionary! { "Font" => dictionary! { "F1" => font_id } });
            let p_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Resources" => r_id,
                "Contents" => c_id,
            });
            kids.push(Object::Reference(p_id));
        }
        doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => kids.len() as i64,
                "Kids" => kids,
            }),
        );
        let cat_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => pages_id });
        doc.trailer.set("Root", cat_id);

        let _orphan = doc.add_object(Stream::new(dictionary! {}, vec![99u8; 4096]));
        let mut raw_pdf = Vec::new();
        doc.save_to(&mut raw_pdf).expect("save text heavy pdf");

        let summary = inspect_pdf(&raw_pdf).expect("inspect zero image heavy pdf");
        assert_eq!(summary.compressible_image_count, 0);
        assert!(summary.gc_removed_objects >= 1);
        assert!(summary.dedup_merged_objects >= 10);
        assert!(summary.dedup_flate_stream_bytes > 0);
        assert!(summary.structural_dict_bytes > 0);

        let est = summary.estimate_compressed_bytes(&CompressionOptions::default());
        let (out, _stats) =
            compress_pdf(&raw_pdf, &CompressionOptions::default(), |_| {}).expect("compress");
        assert!(est < raw_pdf.len() / 2, "expected huge predicted savings: {est} vs {}", raw_pdf.len());
        assert!(out.len() < raw_pdf.len() / 2);
        let ratio = (est as f64) / (out.len() as f64);
        assert!(
            (0.4..=2.2).contains(&ratio),
            "0-image estimate {est} should track actual {} cleanly (ratio {ratio:.2})",
            out.len()
        );
    }

    #[test]
    fn jpx_and_png_formats_encode_decode_estimate_and_solve_cleanly() {
        assert_eq!(
            ImageFormatChoice::ALL,
            [
                ImageFormatChoice::Jpeg,
                ImageFormatChoice::Jpeg2000,
                ImageFormatChoice::Png,
            ]
        );
        assert_eq!(ImageFormatChoice::Jpeg.label(), "JPG");
        assert_eq!(ImageFormatChoice::Jpeg2000.label(), "JPEG 2000");
        assert_eq!(ImageFormatChoice::Png.label(), "PNG");
        assert!(ImageFormatChoice::Jpeg.uses_quality());
        assert!(ImageFormatChoice::Jpeg2000.uses_quality());
        assert!(!ImageFormatChoice::Png.uses_quality());

        let (w, h) = (256u32, 192u32);
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                let fx = x as f64 / w as f64;
                let fy = y as f64 / h as f64;
                rgb[idx] = (32.0 + 185.0 * fx + 24.0 * (fy * 10.0).sin()) as u8;
                rgb[idx + 1] = (28.0 + 165.0 * fy + 26.0 * (fx * 9.0).cos()) as u8;
                rgb[idx + 2] = (210.0 - 150.0 * fx * fy) as u8;
            }
        }
        let rgb_pdf = build_test_pdf_with_image(w, h, "DeviceRGB", rgb.clone(), false);
        let rgb_summary = inspect_pdf(&rgb_pdf).expect("inspect rgb pdf");

        // 1. JPEG 2000: monotone quality & scale, native /JPXDecode, strict hayro decode.
        let mut jpx_ests = Vec::new();
        let mut jpx_sizes = Vec::new();
        for q in [20u8, 50, 85] {
            let opts = CompressionOptions {
                format: ImageFormatChoice::Jpeg2000,
                quality: q,
                scale: 1.00,
                resize_filter: ResizeFilterChoice::Lanczos3,
            };
            let est = rgb_summary.estimate_compressed_bytes(&opts);
            assert!(
                est > rgb_summary.fixed_bytes && est <= rgb_summary.file_bytes,
                "JPX Q{q} estimate {est} out of sane bounds ({}..={})",
                rgb_summary.fixed_bytes,
                rgb_summary.file_bytes
            );
            jpx_ests.push(est);

            let (out, stats) = compress_pdf(&rgb_pdf, &opts, |_| {}).expect("jpx compress");
            assert_eq!(stats.images_recompressed, 1);

            let doc = Document::load_mem(&out).expect("load jpx pdf");
            let img_stream = doc
                .objects
                .values()
                .filter_map(|o| match o {
                    Object::Stream(s) if is_image_xobject(&doc, s) => Some(s),
                    _ => None,
                })
                .next()
                .expect("jpx image stream");
            assert_eq!(
                img_stream.dict.get(b"Filter").unwrap().as_name().unwrap(),
                b"JPXDecode"
            );
            assert_eq!(
                img_stream.dict.get(b"ColorSpace").unwrap().as_name().unwrap(),
                b"DeviceRGB"
            );
            assert_eq!(
                img_stream.dict.get(b"BitsPerComponent").unwrap().as_i64().unwrap(),
                8
            );
            assert!(img_stream.dict.get(b"DecodeParms").is_err());

            let jpx_dec = JpxImage::new(
                &img_stream.content,
                &JpxDecodeSettings {
                    resolve_palette_indices: true,
                    strict: true,
                    target_resolution: None,
                },
            )
            .expect("strict hayro jpx decode");
            assert_eq!((jpx_dec.width(), jpx_dec.height()), (w, h));

            // Ensure our own inspection/compression pipeline also accepts emitted JPX PDFs cleanly.
            let reinspect = inspect_pdf(&out).expect("reinspect jpx pdf");
            assert_eq!(reinspect.compressible_image_count, 1);
            jpx_sizes.push(img_stream.content.len());
        }
        assert!(jpx_ests[0] < jpx_ests[1] && jpx_ests[1] < jpx_ests[2]);
        assert!(jpx_sizes[0] < jpx_sizes[1] && jpx_sizes[1] < jpx_sizes[2]);

        let jpeg_q50_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 50,
            scale: 1.00,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let jpeg_q50_bytes = rgb_summary.estimate_compressed_bytes(&jpeg_q50_opts);
        let jpx_q50_est = rgb_summary.estimate_compressed_bytes(&CompressionOptions {
            format: ImageFormatChoice::Jpeg2000,
            ..jpeg_q50_opts.clone()
        });
        assert!(
            jpx_q50_est < jpeg_q50_bytes,
            "JPX Q50 estimate ({jpx_q50_est}) should be more efficient than JPEG ({jpeg_q50_bytes})"
        );
        let (jpeg_q50_out, _) = compress_pdf(&rgb_pdf, &jpeg_q50_opts, |_| {}).unwrap();
        let (jpx_q50_out, _) = compress_pdf(
            &rgb_pdf,
            &CompressionOptions {
                format: ImageFormatChoice::Jpeg2000,
                ..jpeg_q50_opts.clone()
            },
            |_| {},
        )
        .unwrap();
        assert!(
            jpx_q50_out.len() < jpeg_q50_out.len(),
            "JPX Q50 actual ({}) should beat JPEG Q50 ({}) on smooth RGB",
            jpx_q50_out.len(),
            jpeg_q50_out.len()
        );

        let jpx_half_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg2000,
            quality: 50,
            scale: 0.50,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let (jpx_half_out, _) = compress_pdf(&rgb_pdf, &jpx_half_opts, |_| {}).unwrap();
        assert!(jpx_half_out.len() < jpx_q50_out.len());

        // 2. Round-trip grayscale through PNG (/FlateDecode + /Predictor 15) & JPEG 2000 cleanly.
        let gray_w = 192u32;
        let gray_h = 160u32;
        let mut gray = Vec::with_capacity((gray_w * gray_h) as usize);
        for y in 0..gray_h {
            for x in 0..gray_w {
                let fx = x as f64 / gray_w as f64;
                let fy = y as f64 / gray_h as f64;
                gray.push((28.0 + 185.0 * fx + 28.0 * (fy * 8.0).sin()) as u8);
            }
        }
        let gray_pdf = build_test_pdf_with_image(gray_w, gray_h, "DeviceGray", gray, false);

        for (src_pdf, expected_cs, is_gray) in [
            (&rgb_pdf, b"DeviceRGB".as_slice(), false),
            (&gray_pdf, b"DeviceGray".as_slice(), true),
        ] {
            let summary = inspect_pdf(src_pdf).expect("inspect png source");
            let png_opts = CompressionOptions {
                format: ImageFormatChoice::Png,
                quality: 25,
                scale: 1.00,
                resize_filter: ResizeFilterChoice::Lanczos3,
            };
            let est = summary.estimate_compressed_bytes(&png_opts);
            let (out, stats) = compress_pdf(src_pdf, &png_opts, |_| {}).expect("png compress");
            assert_eq!(stats.images_recompressed, 1);

            let ratio = (est as f64) / (out.len() as f64);
            assert!(
                (0.4..=2.2).contains(&ratio),
                "PNG estimate {est} vs actual {} out of range ({ratio:.2})",
                out.len()
            );

            let doc = Document::load_mem(&out).expect("load png pdf");
            let img_stream = doc
                .objects
                .values()
                .filter_map(|o| match o {
                    Object::Stream(s) if is_image_xobject(&doc, s) => Some(s),
                    _ => None,
                })
                .next()
                .expect("png image stream");
            assert_eq!(
                img_stream.dict.get(b"Filter").unwrap().as_name().unwrap(),
                b"FlateDecode"
            );
            assert_eq!(
                img_stream.dict.get(b"ColorSpace").unwrap().as_name().unwrap(),
                expected_cs
            );

            if is_gray {
                let jpx_g_opts = CompressionOptions {
                    format: ImageFormatChoice::Jpeg2000,
                    quality: 80,
                    scale: 1.00,
                    resize_filter: ResizeFilterChoice::Lanczos3,
                };
                let (out_jg, stats_jg) = compress_pdf(src_pdf, &jpx_g_opts, |_| {}).unwrap();
                assert_eq!(stats_jg.images_recompressed, 1);
                let doc_jg = Document::load_mem(&out_jg).unwrap();
                let s_jg = doc_jg
                    .objects
                    .values()
                    .filter_map(|o| match o {
                        Object::Stream(s) if is_image_xobject(&doc_jg, s) => Some(s),
                        _ => None,
                    })
                    .next()
                    .unwrap();
                assert_eq!(
                    s_jg.dict.get(b"Filter").unwrap().as_name().unwrap(),
                    b"JPXDecode"
                );
                assert_eq!(
                    s_jg.dict.get(b"ColorSpace").unwrap().as_name().unwrap(),
                    b"DeviceGray"
                );
                let dec_jg = JpxImage::new(
                    &s_jg.content,
                    &JpxDecodeSettings {
                        resolve_palette_indices: true,
                        strict: true,
                        target_resolution: None,
                    },
                )
                .expect("strict gray jpx");
                assert_eq!((dec_jg.width(), dec_jg.height()), (gray_w, gray_h));
            }
        }

        // 3. Target-size solver across JPEG 2000 (scale first, then quality) and PNG (scale only, quality untouched).
        let synth = PdfSummary::synthetic_for_test(
            10,
            10 * 1024 * 1024,
            200 * 1024,
            200,
            400,
            48_000,
            false,
        );
        let mut png_solve = CompressionOptions {
            format: ImageFormatChoice::Png,
            quality: 42,
            scale: 1.00,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        synth.apply_target_scale_first(&mut png_solve, 3 * 1024 * 1024);
        assert!(png_solve.scale < 0.99);
        assert_eq!(png_solve.quality, 42, "lossless PNG must not mutate quality");

        let mut jpx_solve = CompressionOptions {
            format: ImageFormatChoice::Jpeg2000,
            quality: 65,
            scale: 1.00,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        synth.apply_target_scale_first(&mut jpx_solve, 3 * 1024 * 1024);
        assert!(jpx_solve.scale > 0.10 && jpx_solve.scale < 1.00);
        assert_eq!(jpx_solve.quality, 65);
    }

    #[test]
    fn calibrated_instant_estimator_tracks_scans_and_solves_targets_cleanly() {
        // Build a 4-page grayscale document and verify instant O(1) estimation & solvers.
        let (w, h) = (1600u32, 1600u32);
        let mut base_doc = Document::with_version("1.5");
        let pages_id = base_doc.new_object_id();
        let mut page_ids = Vec::new();
        for p in 0..4u32 {
            let mut gray = Vec::with_capacity((w * h) as usize);
            let mut rng = 0x9e37_79b9u32 ^ p.wrapping_mul(2654435761);
            for _ in 0..(w * h) {
                rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
                gray.push(245 + ((rng >> 29) as u8));
            }
            for row in 0..22 {
                let y0 = 110 + row * 62 + (p * 3) as usize;
                if y0 + 2 >= h as usize {
                    continue;
                }
                let x_end = (1420usize).min(w as usize - 10);
                for x in 100..x_end {
                    let wave = ((x as f64 * 0.11 + (row as f64) * 0.7 + p as f64).sin() * 9.0) as isize;
                    let py = (y0 as isize + wave).clamp(0, h as isize - 3) as usize;
                    gray[py * (w as usize) + x] = 24;
                    gray[(py + 1) * (w as usize) + x] = 38;
                    gray[(py + 2) * (w as usize) + x] = 85;
                }
            }

            let mut zlib = ZlibEncoder::new(Vec::new(), Compression::best());
            zlib.write_all(&gray).expect("zlib write");
            let flate_bytes = zlib.finish().expect("zlib finish");
            let img_id = base_doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => w as i64,
                    "Height" => h as i64,
                    "ColorSpace" => "DeviceGray",
                    "BitsPerComponent" => 8,
                    "Filter" => "FlateDecode",
                    "Length" => flate_bytes.len() as i64,
                },
                flate_bytes,
            ));
            let res_id = base_doc.add_object(dictionary! {
                "XObject" => dictionary! { "Im0" => img_id },
            });
            let content = Stream::new(
                dictionary! {},
                b"q 595 0 0 842 0 0 cm /Im0 Do Q".to_vec(),
            );
            let cont_id = base_doc.add_object(content);
            let page_id = base_doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 595.into(), 842.into()],
                "Resources" => res_id,
                "Contents" => cont_id,
            });
            page_ids.push(Object::Reference(page_id));
        }
        base_doc.objects.insert(
            pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => page_ids,
                "Count" => 4,
            }),
        );
        let catalog_id = base_doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => pages_id,
        });
        base_doc.trailer.set("Root", catalog_id);
        let mut scan_pdf = Vec::new();
        base_doc.save_to(&mut scan_pdf).expect("save scan pdf");

        let scan_summary = inspect_pdf(&scan_pdf).expect("inspect scan pdf");
        assert_eq!(scan_summary.compressible_image_count, 4);
        assert_eq!(scan_summary.compressible_gray_pixels, 4 * (w as u64) * (h as u64));
        assert!(scan_summary.sample_rgb_jpeg_q50_bpp > 0.08);
        assert!(scan_summary.sample_rgb_flate_bpp > 0.35);

        for (format, quality, scale) in [
            (ImageFormatChoice::Jpeg, 60u8, 1.00f32),
            (ImageFormatChoice::Jpeg, 85, 0.70),
            (ImageFormatChoice::Jpeg2000, 60, 1.00),
            (ImageFormatChoice::Png, 100, 0.80),
        ] {
            let opts = CompressionOptions {
                format,
                quality,
                scale,
                resize_filter: ResizeFilterChoice::Lanczos3,
            };
            let est = scan_summary.estimate_compressed_bytes(&opts) as f64;
            let (out, stats) = compress_pdf(&scan_pdf, &opts, |_| {}).expect("compress scan");
            assert_eq!(stats.images_recompressed, 4, "{format:?} should recompress all 4 scan pages");
            let actual = out.len() as f64;
            let ratio = est / actual;
            assert!(
                (0.4..=2.2).contains(&ratio),
                "{format:?} Q{quality} S{scale:.2}: est {est:.0} vs actual {actual:.0} (ratio {ratio:.3})",
            );
        }

        // Dense 6-image photographic PDF: verify instant solvers respond cleanly without overshoot on estimate.
        let (pw, ph) = (768u32, 768u32);
        let mut photo_doc = Document::with_version("1.5");
        let photo_pages_id = photo_doc.new_object_id();
        let mut photo_page_ids = Vec::new();
        for p in 0..6u32 {
            let mut photo_rgb = Vec::with_capacity((pw * ph * 3) as usize);
            let mut state = 0x1234_5678u32 ^ p.wrapping_mul(747796405);
            for y in 0..ph {
                for x in 0..pw {
                    state = state.wrapping_mul(1664525).wrapping_add(1013904223);
                    let grain = ((((state >> 29) as i32) - 4) * 2) as f64;
                    let fx = (x as f64) / (pw as f64);
                    let fy = (y as f64) / (ph as f64);
                    let bx = (x / 3) as f64;
                    let by = (y / 3) as f64;
                    let tex = (((bx * 1.3).sin() * (by * 1.7).cos()) * 55.0)
                        + (((bx + by) * 0.9).sin() * 35.0);
                    let shade = 40.0 * (fx * 11.0 - fy * 13.0 + p as f64).cos();
                    let luma = 128.0 + tex + shade + grain;
                    let r = (luma + 22.0 * (fx * 11.0).cos()).clamp(0.0, 255.0) as u8;
                    let g = (luma - 14.0 * (fy * 9.0).sin()).clamp(0.0, 255.0) as u8;
                    let b = (luma + 20.0 * ((fx + fy) * 7.0).sin()).clamp(0.0, 255.0) as u8;
                    photo_rgb.push(r);
                    photo_rgb.push(g);
                    photo_rgb.push(b);
                }
            }
            let img_id = photo_doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => pw as i64,
                    "Height" => ph as i64,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                    "Length" => photo_rgb.len() as i64,
                },
                photo_rgb,
            ));
            let res_id = photo_doc.add_object(dictionary! {
                "XObject" => dictionary! { "Im0" => img_id },
            });
            let cont_id = photo_doc.add_object(Stream::new(
                dictionary! {},
                b"q 612 0 0 792 0 0 cm /Im0 Do Q".to_vec(),
            ));
            let page_id = photo_doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => photo_pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
                "Resources" => res_id,
                "Contents" => cont_id,
            });
            photo_page_ids.push(Object::Reference(page_id));
        }
        photo_doc.objects.insert(
            photo_pages_id,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Kids" => photo_page_ids,
                "Count" => 6,
            }),
        );
        let photo_catalog = photo_doc.add_object(dictionary! {
            "Type" => "Catalog",
            "Pages" => photo_pages_id,
        });
        photo_doc.trailer.set("Root", photo_catalog);
        let mut photo_pdf = Vec::new();
        photo_doc.save_to(&mut photo_pdf).expect("save photo pdf");

        let photo_summary = inspect_pdf(&photo_pdf).expect("inspect photo pdf");
        assert_eq!(photo_summary.compressible_image_count, 6);

        let baseline_opts = CompressionOptions {
            format: ImageFormatChoice::Jpeg,
            quality: 75,
            scale: 1.00,
            resize_filter: ResizeFilterChoice::Lanczos3,
        };
        let baseline_est = photo_summary.estimate_compressed_bytes(&baseline_opts);

        let target_75 = (baseline_est * 75) / 100;
        let mut solved_q = baseline_opts.clone();
        solved_q.quality = photo_summary.solve_quality_for_target(&solved_q, target_75);
        assert!(solved_q.quality < 75);
        assert!(photo_summary.estimate_compressed_bytes(&solved_q) <= target_75);

        let target_55 = (baseline_est * 55) / 100;
        let mut solved_scale = baseline_opts;
        photo_summary.apply_target_scale_first(&mut solved_scale, target_55);
        assert!(solved_scale.scale < 1.00);
        assert!(photo_summary.estimate_compressed_bytes(&solved_scale) <= target_55);
    }
}
