//! JPEG encoder and decoder powered by MozJPEG (trellis quantization, progressive
//! scan optimization, overshoot deringing, and adaptive chroma subsampling).

use image::{DynamicImage, GrayImage, RgbImage};
use mozjpeg::{ColorSpace, Compress, Decompress, Format, Marker, ScanMode};

const MAX_JPEG_DIM: usize = 65535;
const CHROMA_444_QUALITY_THRESHOLD: u8 = 85;

#[cold]
fn cold_jpeg_err(msg: String) -> String {
    msg
}

fn validate_dims(width: u32, height: u32, channels: usize, len: usize) -> Result<(), String> {
    if width == 0 || (width as usize) > MAX_JPEG_DIM {
        return Err(cold_jpeg_err(format!(
            "Unsupported JPEG width {width} (must be 1..={MAX_JPEG_DIM})"
        )));
    }
    if height == 0 || (height as usize) > MAX_JPEG_DIM {
        return Err(cold_jpeg_err(format!(
            "Unsupported JPEG height {height} (must be 1..={MAX_JPEG_DIM})"
        )));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .and_then(|px| px.checked_mul(channels))
        .ok_or_else(|| cold_jpeg_err("JPEG pixel buffer size overflow".to_string()))?;
    if len != expected {
        return Err(cold_jpeg_err(format!(
            "Invalid JPEG buffer length: expected {expected} bytes for {width}x{height}x{channels}, got {len}"
        )));
    }
    Ok(())
}

#[cold]
fn panic_payload_message(payload: Box<dyn std::any::Any + Send>) -> String {
    payload
        .downcast_ref::<&str>()
        .copied()
        .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
        .unwrap_or("unknown panic")
        .to_string()
}

fn run_mozjpeg_pass(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    quality: u8,
    progressive: bool,
    optimize_scans: bool,
) -> Result<Vec<u8>, String> {
    // MozJPEG reports internal libjpeg errors by unwinding through C frames.
    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<u8>, String> {
            let mut comp = Compress::new(color_space);
            // Must come first: set_scan_optimization_mode resets defaults to JCP_MAX_COMPRESSION
            // (trellis quantization + overshoot deringing + progressive scan optimization).
            if optimize_scans {
                comp.set_scan_optimization_mode(ScanMode::AllComponentsTogether);
            }
            comp.set_size(width as usize, height as usize);
            comp.set_quality(f32::from(quality));
            comp.set_optimize_scans(optimize_scans);
            if !optimize_scans {
                comp.set_use_scans_in_trellis(false);
            }
            if progressive {
                comp.set_progressive_mode();
            }
            comp.set_optimize_coding(true);

            if color_space == ColorSpace::JCS_RGB {
                if quality <= CHROMA_444_QUALITY_THRESHOLD {
                    comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2)); // 4:2:0
                } else {
                    comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1)); // 4:4:4
                }
            }

            let mut started = comp
                .start_compress(Vec::new())
                .map_err(|e| format!("MozJPEG start_compress failed: {e}"))?;
            started
                .write_scanlines(pixels)
                .map_err(|e| format!("MozJPEG write_scanlines failed: {e}"))?;
            started
                .finish()
                .map_err(|e| format!("MozJPEG finish failed: {e}"))
        }));

    match result {
        Ok(inner) => inner,
        Err(payload) => Err(format!(
            "MozJPEG panicked during encoding: {}",
            panic_payload_message(payload)
        )),
    }
}

fn encode_mozjpeg(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    channels: usize,
    quality: u8,
) -> Result<Vec<u8>, String> {
    validate_dims(width, height, channels, pixels.len())?;
    let clamped_q = quality.clamp(1, 100);

    // Try progressive scan optimization first; fall back without scan optimization or
    // progressive scans if MozJPEG hits an internal Huffman table error on sparse inputs.
    run_mozjpeg_pass(pixels, width, height, color_space, clamped_q, true, true)
        .or_else(|_| run_mozjpeg_pass(pixels, width, height, color_space, clamped_q, true, false))
        .or_else(|_| {
            run_mozjpeg_pass(pixels, width, height, color_space, clamped_q, false, false)
        })
}

/// Encodes an 8-bit grayscale (`L8`) buffer as a MozJPEG stream with trellis
/// quantization and progressive scan optimization.
pub fn encode_jpeg_gray(
    gray: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    encode_mozjpeg(gray, width, height, ColorSpace::JCS_GRAYSCALE, 1, quality)
}

/// Encodes an 8-bit RGB (`Rgb8`) buffer as a MozJPEG stream using `4:2:0` chroma
/// subsampling for `quality <= 85` and `4:4:4` for `quality > 85`, with trellis
/// quantization and progressive scan optimization.
pub fn encode_jpeg_rgb(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    encode_mozjpeg(rgb, width, height, ColorSpace::JCS_RGB, 3, quality)
}

fn encode_mozjpeg_fast_preview(
    pixels: &[u8],
    width: u32,
    height: u32,
    color_space: ColorSpace,
    channels: usize,
    quality: u8,
) -> Result<Vec<u8>, String> {
    validate_dims(width, height, channels, pixels.len())?;
    let clamped_q = quality.clamp(1, 100);

    let result =
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| -> Result<Vec<u8>, String> {
            let mut comp = Compress::new(color_space);
            comp.set_fastest_defaults();
            comp.set_size(width as usize, height as usize);
            comp.set_quality(f32::from(clamped_q));
            comp.set_optimize_scans(false);
            comp.set_optimize_coding(true);

            if color_space == ColorSpace::JCS_RGB {
                if clamped_q <= CHROMA_444_QUALITY_THRESHOLD {
                    comp.set_chroma_sampling_pixel_sizes((2, 2), (2, 2)); // 4:2:0
                } else {
                    comp.set_chroma_sampling_pixel_sizes((1, 1), (1, 1)); // 4:4:4
                }
            }

            let mut started = comp
                .start_compress(Vec::new())
                .map_err(|e| format!("MozJPEG fast preview start_compress failed: {e}"))?;
            started
                .write_scanlines(pixels)
                .map_err(|e| format!("MozJPEG fast preview write_scanlines failed: {e}"))?;
            started
                .finish()
                .map_err(|e| format!("MozJPEG fast preview finish failed: {e}"))
        }));

    match result {
        Ok(inner) => inner,
        Err(payload) => Err(format!(
            "MozJPEG panicked during fast preview encoding: {}",
            panic_payload_message(payload)
        )),
    }
    .or_else(|_| encode_mozjpeg(pixels, width, height, color_space, channels, clamped_q))
}

/// Fast single-pass grayscale JPEG encoder for interactive preview updates.
pub fn encode_jpeg_gray_fast_preview(
    gray: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    encode_mozjpeg_fast_preview(gray, width, height, ColorSpace::JCS_GRAYSCALE, 1, quality)
}

/// Fast single-pass RGB JPEG encoder for interactive preview updates.
pub fn encode_jpeg_rgb_fast_preview(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    encode_mozjpeg_fast_preview(rgb, width, height, ColorSpace::JCS_RGB, 3, quality)
}

fn decode_cmyk_to_rgb(
    cmyk: &[u8],
    width: u32,
    height: u32,
    adobe_inverted: bool,
) -> Result<DynamicImage, String> {
    let pixel_count = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| "Decoded JPEG pixel count overflow".to_string())?;
    let expected = pixel_count
        .checked_mul(4)
        .ok_or_else(|| "Decoded JPEG CMYK buffer size overflow".to_string())?;
    if cmyk.len() != expected {
        return Err(format!(
            "Invalid decoded JPEG CMYK length: expected {expected}, got {}",
            cmyk.len()
        ));
    }

    let src = &cmyk[..expected];
    let mut rgb = vec![0u8; pixel_count * 3];
    if adobe_inverted {
        src.as_chunks::<4>()
            .0
            .iter()
            .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
            .for_each(|(px, dst)| {
                let k = px[3] as u32;
                dst[0] = ((px[0] as u32 * k + 127) / 255) as u8;
                dst[1] = ((px[1] as u32 * k + 127) / 255) as u8;
                dst[2] = ((px[2] as u32 * k + 127) / 255) as u8;
            });
    } else {
        src.as_chunks::<4>()
            .0
            .iter()
            .zip(rgb.as_chunks_mut::<3>().0.iter_mut())
            .for_each(|(px, dst)| {
                let inv_k = 255 - (px[3] as u32);
                dst[0] = (((255 - (px[0] as u32)) * inv_k + 127) / 255) as u8;
                dst[1] = (((255 - (px[1] as u32)) * inv_k + 127) / 255) as u8;
                dst[2] = (((255 - (px[2] as u32)) * inv_k + 127) / 255) as u8;
            });
    }

    RgbImage::from_raw(width, height, rgb)
        .map(DynamicImage::ImageRgb8)
        .ok_or_else(|| "Failed to construct RGB image from CMYK JPEG".to_string())
}

/// Decodes a JPEG stream (`/DCTDecode`) via MozJPEG into `ImageLuma8` or `ImageRgb8`.
pub fn decode_jpeg_to_image(jpeg_bytes: &[u8]) -> Result<DynamicImage, String> {
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(
        || -> Result<DynamicImage, String> {
            let dec = Decompress::with_markers(&[Marker::APP(14)])
                .from_mem(jpeg_bytes)
                .map_err(|e| format!("MozJPEG header error: {e}"))?;

            let w_usize = dec.width();
            let h_usize = dec.height();
            let width = u32::try_from(w_usize)
                .ok()
                .filter(|&w| w >= 1 && (w as usize) <= MAX_JPEG_DIM)
                .ok_or_else(|| format!("Unsupported decoded JPEG width {w_usize}"))?;
            let height = u32::try_from(h_usize)
                .ok()
                .filter(|&h| h >= 1 && (h as usize) <= MAX_JPEG_DIM)
                .ok_or_else(|| format!("Unsupported decoded JPEG height {h_usize}"))?;

            let has_adobe_app14 = dec
                .markers()
                .any(|m| m.marker == Marker::APP(14) && m.data.starts_with(b"Adobe"));

            match dec
                .image()
                .map_err(|e| format!("MozJPEG start_decompress failed: {e}"))?
            {
                Format::Gray(mut started) => {
                    let samples: Vec<u8> = started
                        .read_scanlines()
                        .map_err(|e| format!("MozJPEG read grayscale scanlines failed: {e}"))?;
                    let _ = started.finish();
                    GrayImage::from_raw(width, height, samples)
                        .map(DynamicImage::ImageLuma8)
                        .ok_or_else(|| "Failed to construct grayscale image from JPEG".to_string())
                }
                Format::RGB(mut started) => {
                    let samples: Vec<u8> = started
                        .read_scanlines()
                        .map_err(|e| format!("MozJPEG read RGB scanlines failed: {e}"))?;
                    let _ = started.finish();
                    RgbImage::from_raw(width, height, samples)
                        .map(DynamicImage::ImageRgb8)
                        .ok_or_else(|| "Failed to construct RGB image from JPEG".to_string())
                }
                Format::CMYK(mut started) => {
                    let samples: Vec<u8> = started
                        .read_scanlines()
                        .map_err(|e| format!("MozJPEG read CMYK scanlines failed: {e}"))?;
                    let _ = started.finish();
                    decode_cmyk_to_rgb(&samples, width, height, has_adobe_app14)
                }
            }
        },
    ));

    match result {
        Ok(inner) => inner,
        Err(payload) => Err(format!(
            "MozJPEG panicked during decoding: {}",
            panic_payload_message(payload)
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find_sof_marker_and_sampling_y(jpeg: &[u8]) -> Option<(u8, u8)> {
        let mut i = 0;
        while i + 1 < jpeg.len() {
            if jpeg[i] != 0xFF {
                i += 1;
                continue;
            }
            while i + 1 < jpeg.len() && jpeg[i + 1] == 0xFF {
                i += 1;
            }
            if i + 3 >= jpeg.len() {
                break;
            }
            let marker = jpeg[i + 1];
            if marker == 0xD9 || marker == 0xDA {
                break;
            }
            if marker == 0x00 || (0xD0..=0xD8).contains(&marker) {
                i += 2;
                continue;
            }
            let seg_len = u16::from_be_bytes([jpeg[i + 2], jpeg[i + 3]]) as usize;
            if (marker == 0xC0 || marker == 0xC2) && i + 11 < jpeg.len() {
                return Some((marker, jpeg[i + 11]));
            }
            i += 2 + seg_len;
        }
        None
    }

    fn mse(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let sum: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as f64) - (y as f64);
                d * d
            })
            .sum();
        sum / (a.len() as f64)
    }

    #[test]
    fn encodes_adaptive_subsampling_420_and_444_with_valid_decode() {
        let mut smooth_rgb = vec![0u8; 97 * 73 * 3];
        for y in 0..73usize {
            for x in 0..97usize {
                let p = (y * 97 + x) * 3;
                smooth_rgb[p] = ((x * 255) / 96) as u8;
                smooth_rgb[p + 1] = ((y * 255) / 72) as u8;
                smooth_rgb[p + 2] = (((x + y) * 255) / 168) as u8;
            }
        }

        // Q85 boundary must still use 4:2:0 (0x22) and progressive SOF2 (0xC2).
        let jpeg_420 = encode_jpeg_rgb(&smooth_rgb, 97, 73, 85).expect("encode 4:2:0");
        assert_eq!(find_sof_marker_and_sampling_y(&jpeg_420), Some((0xC2, 0x22)));
        let dec_420 = decode_jpeg_to_image(&jpeg_420)
            .expect("decode 4:2:0")
            .to_rgb8();
        assert_eq!(dec_420.dimensions(), (97, 73));
        assert!(
            mse(&smooth_rgb, dec_420.as_raw()) < 20.0,
            "smooth 4:2:0 MSE too high: {}",
            mse(&smooth_rgb, dec_420.as_raw())
        );

        // Q86 (> 85) switches cleanly to 4:4:4 (0x11).
        let mut hifreq_rgb = vec![0u8; 97 * 73 * 3];
        for y in 0..73usize {
            for x in 0..97usize {
                let p = (y * 97 + x) * 3;
                hifreq_rgb[p] = ((x * 5 + y * 2) % 256) as u8;
                hifreq_rgb[p + 1] = ((y * 7 + x) % 256) as u8;
                hifreq_rgb[p + 2] = (((x ^ y) * 11) % 256) as u8;
            }
        }

        let jpeg_444 = encode_jpeg_rgb(&hifreq_rgb, 97, 73, 86).expect("encode 4:4:4");
        assert_eq!(find_sof_marker_and_sampling_y(&jpeg_444), Some((0xC2, 0x11)));
        let dec_444 = decode_jpeg_to_image(&jpeg_444)
            .expect("decode 4:4:4")
            .to_rgb8();
        assert_eq!(dec_444.dimensions(), (97, 73));
        let hifreq_mse = mse(&hifreq_rgb, dec_444.as_raw());
        // MozJPEG perceptual quant tables + trellis at Q86 have higher MSE on wrap-around XOR static
        // than IJG at Q90, yet still reconstruct sharp chrominance cleanly at 4:4:4.
        assert!(hifreq_mse < 120.0, "4:4:4 Q86 MSE too high: {hifreq_mse}");

        let jpeg_444_q90 = encode_jpeg_rgb(&hifreq_rgb, 97, 73, 90).expect("encode 4:4:4 Q90");
        assert_eq!(
            find_sof_marker_and_sampling_y(&jpeg_444_q90),
            Some((0xC2, 0x11))
        );
        let dec_444_q90 = decode_jpeg_to_image(&jpeg_444_q90)
            .expect("decode 4:4:4 Q90")
            .to_rgb8();
        let q90_mse = mse(&hifreq_rgb, dec_444_q90.as_raw());
        assert!(q90_mse < hifreq_mse, "Q90 MSE ({q90_mse}) should be below Q86 ({hifreq_mse})");

        // MozJPEG Q75 (4:2:0) decodes cleanly and is strictly smaller than 4:4:4 Q86 / Q90.
        let jpeg_moz = encode_jpeg_rgb(&hifreq_rgb, 97, 73, 75).expect("4:2:0 hifreq");
        let dec_moz = decode_jpeg_to_image(&jpeg_moz)
            .expect("decode Q75")
            .to_rgb8();
        assert_eq!(dec_moz.dimensions(), (97, 73));
        assert!(
            jpeg_moz.len() < jpeg_444.len() && jpeg_moz.len() < jpeg_444_q90.len(),
            "Q75 4:2:0 ({}) should be smaller than 4:4:4 Q86 ({}) and Q90 ({})",
            jpeg_moz.len(),
            jpeg_444.len(),
            jpeg_444_q90.len()
        );
    }

    #[test]
    fn encodes_gray_edge_cases_1x1_solid_and_sharp_edges() {
        let one_px = encode_jpeg_gray(&[128u8], 1, 1, 80).expect("1x1 gray");
        let dec_1x1 = decode_jpeg_to_image(&one_px)
            .expect("decode 1x1")
            .to_luma8();
        assert_eq!(dec_1x1.dimensions(), (1, 1));
        assert!((dec_1x1.get_pixel(0, 0).0[0] as i32 - 128).abs() <= 2);

        let solid = vec![240u8; 64 * 64];
        let enc_solid = encode_jpeg_gray(&solid, 64, 64, 50).expect("solid gray");
        let dec_solid = decode_jpeg_to_image(&enc_solid)
            .expect("decode solid")
            .to_luma8();
        assert!(mse(&solid, dec_solid.as_raw()) < 1.0);

        let mut text_rgb = vec![255u8; 40 * 40 * 3];
        for y in 10..30 {
            for x in 10..30 {
                let p = (y * 40 + x) * 3;
                text_rgb[p] = 200;
                text_rgb[p + 1] = 20;
                text_rgb[p + 2] = 20;
            }
        }
        let enc_sharp = encode_jpeg_rgb(&text_rgb, 40, 40, 95).expect("sharp 4:4:4");
        let dec_sharp = decode_jpeg_to_image(&enc_sharp)
            .expect("decode sharp")
            .to_rgb8();
        assert!(mse(&text_rgb, dec_sharp.as_raw()) < 25.0);
    }

    #[test]
    fn rejects_invalid_dimensions_and_buffer_lengths() {
        assert!(encode_jpeg_rgb(&[], 0, 10, 80).is_err());
        assert!(encode_jpeg_gray(&[0; 10], 4, 4, 80).is_err());
        assert!(encode_jpeg_rgb(&[0; 10], 4, 4, 80).is_err());
        assert!(decode_jpeg_to_image(&[]).is_err());
        assert!(decode_jpeg_to_image(b"not a valid jpeg").is_err());
    }

    #[test]
    fn encodes_1600x1600_sparse_high_contrast_grayscale_q90_without_panic() {
        let w = 1600u32;
        let h = 1600u32;
        // Case A: Mostly flat 250 white blocks (>32,767 blocks) with sparse dark grid lines.
        let mut flat_gray = vec![250u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                if (y % 120 == 20 && (100..1500).contains(&x)) || (x % 200 == 50 && (100..1500).contains(&y)) {
                    flat_gray[(y * w + x) as usize] = 35;
                }
            }
        }
        let enc_flat = encode_jpeg_gray(&flat_gray, w, h, 90).expect("1600x1600 flat+lines gray q90");
        assert!(
            decode_jpeg_to_image(&enc_flat)
                .expect("decode flat+lines gray")
                .to_luma8()
                .dimensions()
                == (w, h)
        );

        // Case B: Noisy white paper (245..252) with sparse high-contrast handwritten-style waves.
        let mut paper_gray = Vec::with_capacity((w * h) as usize);
        let mut rng = 0x9e37_79b9u32;
        for _ in 0..(w * h) {
            rng = rng.wrapping_mul(1664525).wrapping_add(1013904223);
            paper_gray.push(245 + ((rng >> 29) as u8));
        }
        for row in 0..22 {
            let y0 = 110 + row * 62;
            for x in 100..1420usize {
                let wave = ((x as f64 * 0.11 + (row as f64) * 0.7).sin() * 9.0) as isize;
                let py = (y0 as isize + wave).clamp(0, h as isize - 3) as usize;
                paper_gray[py * (w as usize) + x] = 24;
                paper_gray[(py + 1) * (w as usize) + x] = 38;
                paper_gray[(py + 2) * (w as usize) + x] = 85;
            }
        }
        let encoded = encode_jpeg_gray(&paper_gray, w, h, 90).expect("1600x1600 paper gray q90");
        let decoded = decode_jpeg_to_image(&encoded)
            .expect("decode 1600x1600 gray")
            .to_luma8();
        assert_eq!(decoded.dimensions(), (w, h));
    }

    #[test]
    fn encodes_sparse_high_contrast_rgb_blocks_at_q100_q85_q75_without_access_violation() {
        // Guards against Windows LLP64 32-bit __builtin_ctzl(size_t) truncation in jcphuff.c
        // where AC positions 1..32 are zero while positions 33..63 contain non-zero HF energy.
        let w = 256u32;
        let h = 256u32;
        let mut sparse_rgb = vec![255u8; (w * h * 3) as usize];
        for by in (0..h).step_by(8) {
            for bx in (0..w).step_by(8) {
                for dy in 0..8 {
                    for dx in 0..8 {
                        let y = by + dy;
                        let x = bx + dx;
                        let p = ((y * w + x) * 3) as usize;
                        // Checkerboard / diagonal HF pattern localized inside 8x8 blocks on solid areas.
                        if (bx / 8 + by / 8) % 3 == 0 {
                            let val = if ((dx + dy) & 1) == 0 { 255u8 } else { 0u8 };
                            sparse_rgb[p] = val;
                            sparse_rgb[p + 1] = val;
                            sparse_rgb[p + 2] = val;
                        } else if dx == 7 && dy == 7 {
                            sparse_rgb[p] = 16;
                            sparse_rgb[p + 1] = 32;
                            sparse_rgb[p + 2] = 240;
                        }
                    }
                }
            }
        }

        for q in [100u8, 85u8, 75u8] {
            let full = encode_jpeg_rgb(&sparse_rgb, w, h, q).expect("full mozjpeg rgb");
            let dec_full = decode_jpeg_to_image(&full)
                .expect("decode full sparse rgb")
                .to_rgb8();
            assert_eq!(dec_full.dimensions(), (w, h));

            let fast =
                encode_jpeg_rgb_fast_preview(&sparse_rgb, w, h, q).expect("fast preview rgb");
            let dec_fast = decode_jpeg_to_image(&fast)
                .expect("decode fast sparse rgb")
                .to_rgb8();
            assert_eq!(dec_fast.dimensions(), (w, h));
        }
    }
}
