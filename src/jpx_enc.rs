use smallvec::{smallvec, SmallVec};
use std::sync::OnceLock;

/// Level-shifted 2D sample precision used by PDF `/JPXDecode` streams (8 bits unsigned).
const BIT_DEPTH: u8 = 8;
/// Number of guard bits signaled in QCD (`G = 2..=7`) to prevent magnitude overflow across subbands.
const MAX_DECOMP_LEVELS: u8 = 5;
/// Code-block dimension (`2^6 = 64`).
const CB_DIM: u32 = 64;

/// Inverse 9-7 lifting parameters matching ITU-T T.800 Table F.4 (`hayro-jpeg2000` `idwt.rs`).
const NEG_ALPHA: f32 = 1.586_134_3;
const NEG_BETA: f32 = 0.052_980_117;
const NEG_GAMMA: f32 = -0.882_911_1;
const NEG_DELTA: f32 = -0.443_506_87;
const KAPPA: f32 = 1.230_174_1;
const INV_KAPPA: f32 = 1.0 / KAPPA;

/// Chroma quantization multiplier relative to luma when RCT/ICT color decorrelation (`MCT = 1`) is enabled.
const CHROMA_STEP_MULT: f32 = 1.55;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SubBandKind {
    LowLow,
    HighLow,
    LowHigh,
    HighHigh,
}

impl SubBandKind {
    fn log_gain(self) -> u16 {
        match self {
            Self::LowLow => 0,
            Self::HighLow | Self::LowHigh => 1,
            Self::HighHigh => 2,
        }
    }
}

/// Encodes an 8-bit grayscale (`Luma8`) image into an ITU-T T.800 JPEG 2000 Part 1 codestream (`.j2k`).
#[cold]
fn cold_jpx_err(msg: String) -> String {
    msg
}

const HDR10_LEVEL_SHIFT_LUT: [f32; 256] = {
    let mut lut = [0.0f32; 256];
    let mut i = 0usize;
    while i < 256 {
        lut[i] = (((i as u32) * 1023 + 127) / 255) as f32 - 512.0;
        i += 1;
    }
    lut
};

pub fn encode_jpx_gray(gray: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err(cold_jpx_err("JPEG 2000 width and height must be non-zero.".into()));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 dimensions overflow usize.".to_string()))?;
    if gray.len() != expected {
        return Err(cold_jpx_err(format!(
            "Grayscale buffer length {} does not match {width}x{height}.",
            gray.len()
        )));
    }

    let comp0: Vec<f32> = gray[..expected]
        .iter()
        .map(|&p| (p as f32) - 128.0)
        .collect();
    encode_jpx_components(smallvec![comp0], width, height, quality, false, BIT_DEPTH)
}

/// Encodes a 1-bit bi-level grayscale image (`0` = black, `>= 128` = white) into a 1-bit
/// ITU-T T.800 JPEG 2000 Part 1 codestream (`Ssiz = 0`, 1 bpc).
pub fn encode_jpx_bilevel_1bit(
    gray: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err(cold_jpx_err("JPEG 2000 width and height must be non-zero.".into()));
    }
    let expected = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 dimensions overflow usize.".to_string()))?;
    if gray.len() != expected {
        return Err(cold_jpx_err(format!(
            "1-bit grayscale buffer length {} does not match {width}x{height}.",
            gray.len()
        )));
    }

    // For bit_depth = 1, hayro-jpeg2000 / ITU-T T.800 applies level shift 1 << (1 - 1) = 1.0,
    // mapping {0.0, 1.0} <-> {-1.0, 0.0}.
    let comp0: Vec<f32> = gray[..expected]
        .iter()
        .map(|&p| if p >= 128 { 0.0 } else { -1.0 })
        .collect();
    encode_jpx_components(smallvec![comp0], width, height, quality, false, 1)
}

/// Encodes an 8-bit RGB (`Rgb8`) image into an ITU-T T.800 JPEG 2000 Part 1 codestream (`.j2k`)
/// using the Irreversible Color Transform (ICT) + CDF 9/7 DWT + EBCOT arithmetic coding.
pub fn encode_jpx_rgb(rgb: &[u8], width: u32, height: u32, quality: u8) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err(cold_jpx_err("JPEG 2000 width and height must be non-zero.".into()));
    }
    let px = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 dimensions overflow usize.".to_string()))?;
    let expected = px
        .checked_mul(3)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 RGB buffer overflows usize.".to_string()))?;
    if rgb.len() != expected {
        return Err(cold_jpx_err(format!(
            "RGB buffer length {} does not match {width}x{height}x3.",
            rgb.len()
        )));
    }

    let src = &rgb[..expected];
    let mut y = vec![0.0f32; px];
    let mut cb = vec![0.0f32; px];
    let mut cr = vec![0.0f32; px];
    src.as_chunks::<3>()
        .0
        .iter()
        .zip(y.iter_mut().zip(cb.iter_mut().zip(cr.iter_mut())))
        .for_each(|(chunk, (yo, (cbo, cro)))| {
            let r = (chunk[0] as f32) - 128.0;
            let g = (chunk[1] as f32) - 128.0;
            let b = (chunk[2] as f32) - 128.0;
            *yo = 0.299 * r + 0.587 * g + 0.114 * b;
            *cbo = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
            *cro = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        });

    encode_jpx_components(smallvec![y, cb, cr], width, height, quality, true, BIT_DEPTH)
}

/// Encodes an 8-bit RGB (`Rgb8`) image into a 10-bit-per-channel (`30-bit` HDR, `Ssiz = 9`)
/// ITU-T T.800 JPEG 2000 Part 1 codestream using ICT + CDF 9/7 DWT + EBCOT.
pub fn encode_jpx_rgb_10bit(
    rgb: &[u8],
    width: u32,
    height: u32,
    quality: u8,
) -> Result<Vec<u8>, String> {
    if width == 0 || height == 0 {
        return Err(cold_jpx_err("JPEG 2000 width and height must be non-zero.".into()));
    }
    let px = (width as usize)
        .checked_mul(height as usize)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 dimensions overflow usize.".to_string()))?;
    let expected = px
        .checked_mul(3)
        .ok_or_else(|| cold_jpx_err("JPEG 2000 RGB buffer overflows usize.".to_string()))?;
    if rgb.len() != expected {
        return Err(cold_jpx_err(format!(
            "RGB buffer length {} does not match {width}x{height}x3.",
            rgb.len()
        )));
    }

    // Map 0..=255 -> 0..=1023, then subtract 10-bit level shift 512.0 via compile-time LUT.
    let src = &rgb[..expected];
    let mut y = vec![0.0f32; px];
    let mut cb = vec![0.0f32; px];
    let mut cr = vec![0.0f32; px];
    src.as_chunks::<3>()
        .0
        .iter()
        .zip(y.iter_mut().zip(cb.iter_mut().zip(cr.iter_mut())))
        .for_each(|(chunk, (yo, (cbo, cro)))| {
            let r = HDR10_LEVEL_SHIFT_LUT[chunk[0] as usize];
            let g = HDR10_LEVEL_SHIFT_LUT[chunk[1] as usize];
            let b = HDR10_LEVEL_SHIFT_LUT[chunk[2] as usize];
            *yo = 0.299 * r + 0.587 * g + 0.114 * b;
            *cbo = -0.168_736 * r - 0.331_264 * g + 0.5 * b;
            *cro = 0.5 * r - 0.418_688 * g - 0.081_312 * b;
        });

    encode_jpx_components(smallvec![y, cb, cr], width, height, quality, true, 10)
}

/// Computes exact 1D CDF 9/7 synthesis impulse L2 norms `norms.0[d]` (lowpass depth d=0..=5)
/// and `norms.1[d]` (highpass depth d=1..=5) directly from `filter_97i` lifting constants.
fn dwt_97_1d_norms() -> &'static ([f64; 6], [f64; 6]) {
    static NORMS: OnceLock<([f64; 6], [f64; 6])> = OnceLock::new();
    NORMS.get_or_init(|| {
        const BUF_N: usize = 256;
        fn idwt_1d_step(low: &[f64], high: &[f64]) -> Vec<f64> {
            let nl = low.len();
            let nh = high.len();
            let mut l = vec![0.0f64; nl + 2];
            let mut h = vec![0.0f64; nh + 2];
            for i in 0..nl {
                l[i + 1] = low[i] * (KAPPA as f64);
            }
            for i in 0..nh {
                h[i + 1] = high[i] * (INV_KAPPA as f64);
            }
            // lift_low NEG_DELTA (offset = 1 -> h[i], h[i+1])
            h[0] = h[1];
            h[nh + 1] = h[nh];
            for i in 0..nl {
                l[i + 1] += (h[i] + h[i + 1]) * (NEG_DELTA as f64);
            }
            // lift_high NEG_GAMMA (offset = 0 -> l[i+1], l[i+2])
            l[0] = l[1];
            l[nl + 1] = l[nl];
            for i in 0..nh {
                h[i + 1] += (l[i + 1] + l[i + 2]) * (NEG_GAMMA as f64);
            }
            // lift_low NEG_BETA (offset = 1 -> h[i], h[i+1])
            h[0] = h[1];
            h[nh + 1] = h[nh];
            for i in 0..nl {
                l[i + 1] += (h[i] + h[i + 1]) * (NEG_BETA as f64);
            }
            // lift_high NEG_ALPHA (offset = 0 -> l[i+1], l[i+2])
            l[0] = l[1];
            l[nl + 1] = l[nl];
            for i in 0..nh {
                h[i + 1] += (l[i + 1] + l[i + 2]) * (NEG_ALPHA as f64);
            }

            let mut out = vec![0.0f64; nl + nh];
            for i in 0..nl {
                out[i * 2] = l[i + 1];
            }
            for i in 0..nh {
                out[i * 2 + 1] = h[i + 1];
            }
            out
        }

        let mut low_norms = [1.0f64; 6];
        let mut high_norms = [1.0f64; 6];

        // Impulse at center of LL at level d, synthesized up d stages to pixel grid.
        let mut sig_l = vec![0.0f64; BUF_N];
        sig_l[BUF_N / 2] = 1.0;
        let mut cur_l = sig_l;
        for slot in low_norms.iter_mut().skip(1) {
            let z = vec![0.0f64; cur_l.len()];
            let half = idwt_1d_step(&cur_l, &z);
            let norm = half.iter().map(|v| v * v).sum::<f64>().sqrt();
            *slot = norm;
            cur_l = half;
        }

        // Highpass impulse at level 1 -> synthesized d times (stage 1 from highpass, rest LL->pixels).
        let mut sig_h = vec![0.0f64; BUF_N];
        sig_h[BUF_N / 2] = 1.0;
        let zeros = vec![0.0f64; BUF_N];
        let mut cur_h = idwt_1d_step(&zeros, &sig_h);
        high_norms[1] = cur_h.iter().map(|v| v * v).sum::<f64>().sqrt();
        for slot in high_norms.iter_mut().skip(2) {
            let z = vec![0.0f64; cur_h.len()];
            cur_h = idwt_1d_step(&cur_h, &z);
            *slot = cur_h.iter().map(|v| v * v).sum::<f64>().sqrt();
        }

        (low_norms, high_norms)
    })
}

fn subband_l2_norm(kind: SubBandKind, depth_from_finest: u8) -> f64 {
    let (low_norms, high_norms) = dwt_97_1d_norms();
    let d = (depth_from_finest as usize).min(5);
    match kind {
        SubBandKind::LowLow => low_norms[d] * low_norms[d],
        SubBandKind::HighLow => high_norms[d] * low_norms[d],
        SubBandKind::LowHigh => low_norms[d] * high_norms[d],
        SubBandKind::HighHigh => high_norms[d] * high_norms[d],
    }
}

/// Maps UI quality (`1..=100`) to a pixel-domain deadzone quantization base step.
/// Calibrated so Q=100 delivers near-lossless reconstruction (~46+ dB) while Q=10..70
/// produces smaller bitrates than baseline JPEG (~0.65x-0.80x).
fn quality_to_base_step(quality: u8) -> f32 {
    let q = quality.clamp(1, 100) as f64;
    if q >= 100.0 {
        return 1.2;
    }
    // Smooth log-linear curve anchored around JPEG 2000 efficiency targets:
    // Q80 -> ~4.4, Q50 -> ~13.0, Q30 -> ~27.0, Q1 -> ~95.0.
    let t = (100.0 - q) / 99.0;
    (1.2 * (79.0_f64).powf(t.powf(0.88))) as f32
}

struct SubBandData {
    kind: SubBandKind,
    resolution: u8,
    width: u32,
    height: u32,
    coeffs: Vec<f32>,
}

/// Performs 1D CDF 9/7 forward analysis transform on `n >= 2` samples into `low` (`ceil(n/2)`)
/// and `high` (`n/2`), exact algebraic inverse of `hayro-jpeg2000`'s `filter_97i`.
fn dwt_97_1d_forward(input: &[f32], low: &mut [f32], high: &mut [f32]) {
    let n = input.len();
    debug_assert!(n >= 2);
    let nl = n.div_ceil(2);
    let nh = n / 2;
    debug_assert_eq!(low.len(), nl);
    debug_assert_eq!(high.len(), nh);

    for i in 0..nl {
        low[i] = input[i * 2];
    }
    for i in 0..nh {
        high[i] = input[i * 2 + 1];
    }

    for i in 0..nh {
        let right = low[(i + 1).min(nl - 1)];
        high[i] -= NEG_ALPHA * (low[i] + right);
    }
    for i in 0..nl {
        let left = high[i.saturating_sub(1)];
        let right = high[i.min(nh - 1)];
        low[i] -= NEG_BETA * (left + right);
    }
    for i in 0..nh {
        let right = low[(i + 1).min(nl - 1)];
        high[i] -= NEG_GAMMA * (low[i] + right);
    }
    for i in 0..nl {
        let left = high[i.saturating_sub(1)];
        let right = high[i.min(nh - 1)];
        low[i] -= NEG_DELTA * (left + right);
    }

    for v in low.iter_mut() {
        *v *= INV_KAPPA;
    }
    for v in high.iter_mut() {
        *v *= KAPPA;
    }
}

/// Runs forward `N_L`-level 2D CDF 9/7 analysis, returning subbands in ITU-T T.800 order:
/// `[res=0 LL, res=1 (HL, LH, HH), ..., res=N_L (HL, LH, HH)]`.
fn forward_dwt_2d(mut cur_data: Vec<f32>, width: u32, height: u32, n_l: u8) -> Vec<SubBandData> {
    let mut cur_w = width as usize;
    let mut cur_h = height as usize;

    // Collect levels from finest (depth = 1) to deepest (depth = n_l).
    let mut detail_levels: Vec<[SubBandData; 3]> = Vec::with_capacity(n_l as usize);

    let mut col_in = Vec::with_capacity(cur_h);
    let mut col_l = Vec::with_capacity(cur_h.div_ceil(2));
    let mut col_h = Vec::with_capacity(cur_h / 2);
    let mut row_l = Vec::with_capacity(cur_w.div_ceil(2));
    let mut row_h = Vec::with_capacity(cur_w / 2);

    for level_idx in 0..n_l {
        let w_l = cur_w.div_ceil(2);
        let w_h = cur_w / 2;
        let h_l = cur_h.div_ceil(2);
        let h_h = cur_h / 2;

        // Vertical forward 1D pass first (exact reverse of HOR->VER synthesis in idwt.rs).
        let mut l_rows = vec![0.0f32; cur_w * h_l];
        let mut h_rows = vec![0.0f32; cur_w * h_h];
        col_in.resize(cur_h, 0.0);
        col_l.resize(h_l, 0.0);
        col_h.resize(h_h, 0.0);

        for x in 0..cur_w {
            for y in 0..cur_h {
                col_in[y] = cur_data[y * cur_w + x];
            }
            dwt_97_1d_forward(&col_in, &mut col_l, &mut col_h);
            for y in 0..h_l {
                l_rows[y * cur_w + x] = col_l[y];
            }
            for y in 0..h_h {
                h_rows[y * cur_w + x] = col_h[y];
            }
        }

        // Horizontal forward 1D pass across L rows (-> LL, HL) and H rows (-> LH, HH).
        let mut ll = vec![0.0f32; w_l * h_l];
        let mut hl = vec![0.0f32; w_h * h_l];
        let mut lh = vec![0.0f32; w_l * h_h];
        let mut hh = vec![0.0f32; w_h * h_h];

        row_l.resize(w_l, 0.0);
        row_h.resize(w_h, 0.0);

        for y in 0..h_l {
            let row = &l_rows[y * cur_w..(y + 1) * cur_w];
            dwt_97_1d_forward(row, &mut row_l, &mut row_h);
            ll[y * w_l..(y + 1) * w_l].copy_from_slice(&row_l);
            hl[y * w_h..(y + 1) * w_h].copy_from_slice(&row_h);
        }
        for y in 0..h_h {
            let row = &h_rows[y * cur_w..(y + 1) * cur_w];
            dwt_97_1d_forward(row, &mut row_l, &mut row_h);
            lh[y * w_l..(y + 1) * w_l].copy_from_slice(&row_l);
            hh[y * w_h..(y + 1) * w_h].copy_from_slice(&row_h);
        }

        let res = n_l - level_idx;
        detail_levels.push([
            SubBandData {
                kind: SubBandKind::HighLow,
                resolution: res,
                width: w_h as u32,
                height: h_l as u32,
                coeffs: hl,
            },
            SubBandData {
                kind: SubBandKind::LowHigh,
                resolution: res,
                width: w_l as u32,
                height: h_h as u32,
                coeffs: lh,
            },
            SubBandData {
                kind: SubBandKind::HighHigh,
                resolution: res,
                width: w_h as u32,
                height: h_h as u32,
                coeffs: hh,
            },
        ]);

        cur_data = ll;
        cur_w = w_l;
        cur_h = h_l;
    }

    let mut out = Vec::with_capacity(1 + 3 * (n_l as usize));
    out.push(SubBandData {
        kind: SubBandKind::LowLow,
        resolution: 0,
        width: cur_w as u32,
        height: cur_h as u32,
        coeffs: cur_data,
    });
    for level in detail_levels.into_iter().rev() {
        out.extend(level);
    }
    out
}

#[derive(Clone, Copy)]
struct QuantStepParams {
    exponent: u16,
    mantissa: u16,
    delta: f32,
}

fn encode_step_size(target_delta: f32, log_gain: u16, bit_depth: u8) -> QuantStepParams {
    let r_b = (bit_depth as u16) + log_gain;
    let scale_pow = (1u32 << r_b) as f64;
    let clamped = (target_delta as f64).clamp(1.0 / 4096.0, scale_pow * 0.99);
    let val = clamped / scale_pow;
    let mut eps = (-val.log2().floor()) as i32;
    eps = eps.clamp(0, 24);
    let factor = val * 2.0_f64.powi(eps);
    let mut mantissa = ((factor - 1.0) * 2048.0).round() as i32;
    if mantissa >= 2048 {
        mantissa = 0;
        eps -= 1;
    }
    if mantissa < 0 {
        mantissa = 0;
    }
    let eps = eps.clamp(0, 24) as u16;
    let mantissa = mantissa as u16;
    let exact_delta =
        (2.0_f64.powi((r_b as i32) - (eps as i32)) * (1.0 + (mantissa as f64) / 2048.0)) as f32;
    QuantStepParams {
        exponent: eps,
        mantissa,
        delta: exact_delta,
    }
}

/// Quantized coefficient storage (sign in bit 31, magnitude in bits 0..30) matching `bitplane.rs`.
struct QuantizedSubBand {
    kind: SubBandKind,
    width: u32,
    height: u32,
    num_bitplanes: u8,
    mags: Vec<u32>,
    signs: Vec<u8>,
}

fn quantize_component(
    subbands: &[SubBandData],
    n_l: u8,
    base_step: f32,
    bit_depth: u8,
) -> (u8, Vec<QuantStepParams>, Vec<QuantizedSubBand>) {
    let mut steps = Vec::with_capacity(subbands.len());
    let mut max_mags = Vec::with_capacity(subbands.len());
    let mut raw_q = Vec::with_capacity(subbands.len());

    let mut min_guard_needed = 2u8;

    for sb in subbands {
        let depth_from_finest = if sb.resolution == 0 {
            n_l
        } else {
            n_l + 1 - sb.resolution
        };
        let norm = subband_l2_norm(sb.kind, depth_from_finest) as f32;
        // Weight finest HH1 slightly coarser for high-frequency wavelet compression gains.
        let csf_mult = if sb.kind == SubBandKind::HighHigh && depth_from_finest == 1 {
            1.22
        } else {
            1.0
        };
        let target_delta = (base_step * csf_mult) / norm.max(0.25);
        let qstep = encode_step_size(target_delta, sb.kind.log_gain(), bit_depth);
        let inv_delta = 1.0 / qstep.delta;

        let mut mags = Vec::with_capacity(sb.coeffs.len());
        let mut signs = Vec::with_capacity(sb.coeffs.len());
        let mut max_mag = 0u32;

        for &c in &sb.coeffs {
            let q = (c.abs() * inv_delta).floor() as u32;
            if q > max_mag {
                max_mag = q;
            }
            mags.push(q);
            signs.push(if c < 0.0 { 1 } else { 0 });
        }

        if max_mag > 0 {
            let bits_needed = (32 - max_mag.leading_zeros()) as u8;
            let needed_guard = (bits_needed + 1).saturating_sub(qstep.exponent as u8);
            if needed_guard > min_guard_needed {
                min_guard_needed = needed_guard;
            }
        }

        steps.push(qstep);
        max_mags.push(max_mag);
        raw_q.push((mags, signs));
    }

    let guard_bits = min_guard_needed.clamp(2, 7);
    let mut out = Vec::with_capacity(subbands.len());
    for (idx, (sb, (mut mags, signs))) in subbands.iter().zip(raw_q).enumerate() {
        let num_bitplanes = (guard_bits + (steps[idx].exponent as u8) - 1).min(30);
        let max_allowed = if num_bitplanes == 0 {
            0
        } else {
            (1u32 << num_bitplanes) - 1
        };
        if max_mags[idx] > max_allowed {
            for m in &mut mags {
                if *m > max_allowed {
                    *m = max_allowed;
                }
            }
        }
        out.push(QuantizedSubBand {
            kind: sb.kind,
            width: sb.width,
            height: sb.height,
            num_bitplanes,
            mags,
            signs,
        });
    }

    (guard_bits, steps, out)
}

#[derive(Clone, Copy)]
struct QeEntry {
    qe: u32,
    nmps: u8,
    nlps: u8,
    switch: bool,
}

/// Exact ITU-T T.800 Table C.2 state machine matching `hayro-jpeg2000` `QE_TABLE`.
#[rustfmt::skip]
const QE_TABLE: [QeEntry; 47] = [
    QeEntry { qe: 0x5601, nmps: 1,  nlps: 1,  switch: true  },
    QeEntry { qe: 0x3401, nmps: 2,  nlps: 6,  switch: false },
    QeEntry { qe: 0x1801, nmps: 3,  nlps: 9,  switch: false },
    QeEntry { qe: 0x0AC1, nmps: 4,  nlps: 12, switch: false },
    QeEntry { qe: 0x0521, nmps: 5,  nlps: 29, switch: false },
    QeEntry { qe: 0x0221, nmps: 38, nlps: 33, switch: false },
    QeEntry { qe: 0x5601, nmps: 7,  nlps: 6,  switch: true  },
    QeEntry { qe: 0x5401, nmps: 8,  nlps: 14, switch: false },
    QeEntry { qe: 0x4801, nmps: 9,  nlps: 14, switch: false },
    QeEntry { qe: 0x3801, nmps: 10, nlps: 14, switch: false },
    QeEntry { qe: 0x3001, nmps: 11, nlps: 17, switch: false },
    QeEntry { qe: 0x2401, nmps: 12, nlps: 18, switch: false },
    QeEntry { qe: 0x1C01, nmps: 13, nlps: 20, switch: false },
    QeEntry { qe: 0x1601, nmps: 29, nlps: 21, switch: false },
    QeEntry { qe: 0x5601, nmps: 15, nlps: 14, switch: true  },
    QeEntry { qe: 0x5401, nmps: 16, nlps: 14, switch: false },
    QeEntry { qe: 0x5101, nmps: 17, nlps: 15, switch: false },
    QeEntry { qe: 0x4801, nmps: 18, nlps: 16, switch: false },
    QeEntry { qe: 0x3801, nmps: 19, nlps: 17, switch: false },
    QeEntry { qe: 0x3401, nmps: 20, nlps: 18, switch: false },
    QeEntry { qe: 0x3001, nmps: 21, nlps: 19, switch: false },
    QeEntry { qe: 0x2801, nmps: 22, nlps: 19, switch: false },
    QeEntry { qe: 0x2401, nmps: 23, nlps: 20, switch: false },
    QeEntry { qe: 0x2201, nmps: 24, nlps: 21, switch: false },
    QeEntry { qe: 0x1C01, nmps: 25, nlps: 22, switch: false },
    QeEntry { qe: 0x1801, nmps: 26, nlps: 23, switch: false },
    QeEntry { qe: 0x1601, nmps: 27, nlps: 24, switch: false },
    QeEntry { qe: 0x1401, nmps: 28, nlps: 25, switch: false },
    QeEntry { qe: 0x1201, nmps: 29, nlps: 26, switch: false },
    QeEntry { qe: 0x1101, nmps: 30, nlps: 27, switch: false },
    QeEntry { qe: 0x0AC1, nmps: 31, nlps: 28, switch: false },
    QeEntry { qe: 0x09C1, nmps: 32, nlps: 29, switch: false },
    QeEntry { qe: 0x08A1, nmps: 33, nlps: 30, switch: false },
    QeEntry { qe: 0x0521, nmps: 34, nlps: 31, switch: false },
    QeEntry { qe: 0x0441, nmps: 35, nlps: 32, switch: false },
    QeEntry { qe: 0x02A1, nmps: 36, nlps: 33, switch: false },
    QeEntry { qe: 0x0221, nmps: 37, nlps: 34, switch: false },
    QeEntry { qe: 0x0141, nmps: 38, nlps: 35, switch: false },
    QeEntry { qe: 0x0111, nmps: 39, nlps: 36, switch: false },
    QeEntry { qe: 0x0085, nmps: 40, nlps: 37, switch: false },
    QeEntry { qe: 0x0049, nmps: 41, nlps: 38, switch: false },
    QeEntry { qe: 0x0025, nmps: 42, nlps: 39, switch: false },
    QeEntry { qe: 0x0015, nmps: 43, nlps: 40, switch: false },
    QeEntry { qe: 0x0009, nmps: 44, nlps: 41, switch: false },
    QeEntry { qe: 0x0005, nmps: 45, nlps: 42, switch: false },
    QeEntry { qe: 0x0001, nmps: 45, nlps: 43, switch: false },
    QeEntry { qe: 0x5601, nmps: 46, nlps: 46, switch: false },
];

#[derive(Clone, Copy, Default)]
struct MqContext {
    index: u8,
    mps: u32,
}

impl MqContext {
    fn reset_with_index(&mut self, index: u8) {
        self.index = index;
        self.mps = 0;
    }
}

/// ITU-T T.800 Annex C.2 MQ-coder arithmetic encoder.
struct MqEncoder {
    a: u32,
    c: u32,
    ct: u32,
    buf: Vec<u8>,
    bp: usize,
}

impl MqEncoder {
    fn new() -> Self {
        let mut buf = Vec::with_capacity(256);
        buf.push(0);
        Self {
            a: 0x8000,
            c: 0,
            ct: 12,
            buf,
            bp: 0,
        }
    }

    #[inline(always)]
    fn encode_bit(&mut self, ctx: &mut MqContext, bit: u32) {
        let entry = &QE_TABLE[ctx.index as usize];
        self.a -= entry.qe;
        if (bit & 1) == ctx.mps {
            if (self.a & 0x8000) == 0 {
                if self.a < entry.qe {
                    self.a = entry.qe;
                } else {
                    self.c += entry.qe;
                }
                ctx.index = entry.nmps;
                self.renorme();
            } else {
                self.c += entry.qe;
            }
        } else {
            if self.a < entry.qe {
                self.c += entry.qe;
            } else {
                self.a = entry.qe;
            }
            if entry.switch {
                ctx.mps ^= 1;
            }
            ctx.index = entry.nlps;
            self.renorme();
        }
    }

    #[inline(always)]
    fn renorme(&mut self) {
        loop {
            self.a <<= 1;
            self.c <<= 1;
            self.ct -= 1;
            if self.ct == 0 {
                self.byte_out();
            }
            if (self.a & 0x8000) != 0 {
                break;
            }
        }
    }

    #[inline(always)]
    fn byte_out(&mut self) {
        if self.buf[self.bp] == 0xFF {
            self.buf.push(((self.c >> 20) & 0xFF) as u8);
            self.bp += 1;
            self.c &= 0x000F_FFFF;
            self.ct = 7;
        } else if (self.c & 0x0800_0000) == 0 {
            self.buf.push(((self.c >> 19) & 0xFF) as u8);
            self.bp += 1;
            self.c &= 0x0007_FFFF;
            self.ct = 8;
        } else {
            self.buf[self.bp] = self.buf[self.bp].wrapping_add(1);
            if self.buf[self.bp] == 0xFF {
                self.c &= 0x07FF_FFFF;
                self.buf.push(((self.c >> 20) & 0xFF) as u8);
                self.bp += 1;
                self.c &= 0x000F_FFFF;
                self.ct = 7;
            } else {
                self.buf.push(((self.c >> 19) & 0xFF) as u8);
                self.bp += 1;
                self.c &= 0x0007_FFFF;
                self.ct = 8;
            }
        }
    }

    fn flush(mut self) -> Vec<u8> {
        let tempc = self.c + self.a;
        self.c |= 0x0000_FFFF;
        if self.c >= tempc {
            self.c -= 0x8000;
        }
        self.c <<= self.ct;
        self.byte_out();
        self.c <<= self.ct;
        self.byte_out();

        let mut payload = self.buf.split_off(1);
        while payload.last() == Some(&0xFF) {
            payload.pop();
        }
        payload
    }
}

/// Bit-packed sign contexts matching ITU-T T.800 Table D.2 / `hayro-jpeg2000` `SIGN_CONTEXT_LOOKUP`.
#[rustfmt::skip]
const SIGN_CONTEXT_LOOKUP: [(u8, u8); 256] = [
    (9,0), (10,0), (10,1), (0,0), (12,0), (13,0), (11,0), (0,0), (12,1), (11,1),
    (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (12,0), (13,0), (11,0), (0,0),
    (12,0), (13,0), (11,0), (0,0), (9,0), (10,0), (10,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (12,1), (11,1), (13,1), (0,0), (9,0), (10,0), (10,1), (0,0),
    (12,1), (11,1), (13,1), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (10,0), (10,0), (9,0), (0,0), (13,0), (13,0), (12,0),
    (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,0),
    (13,0), (12,0), (0,0), (13,0), (13,0), (12,0), (0,0), (10,0), (10,0), (9,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (11,1), (11,1), (12,1), (0,0), (10,0),
    (10,0), (9,0), (0,0), (11,1), (11,1), (12,1), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (10,1), (9,0), (10,1), (0,0),
    (11,0), (12,0), (11,0), (0,0), (13,1), (12,1), (13,1), (0,0), (0,0), (0,0),
    (0,0), (0,0), (11,0), (12,0), (11,0), (0,0), (11,0), (12,0), (11,0), (0,0),
    (10,1), (9,0), (10,1), (0,0), (0,0), (0,0), (0,0), (0,0), (13,1), (12,1),
    (13,1), (0,0), (10,1), (9,0), (10,1), (0,0), (13,1), (12,1), (13,1), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
    (0,0), (0,0), (0,0), (0,0), (0,0), (0,0), (0,0),
];

#[rustfmt::skip]
const ZERO_CTX_LL_LH_LOOKUP: [u8; 256] = [
    0, 3, 1, 3, 5, 7, 6, 7, 1, 3, 2, 3, 6, 7, 6, 7, 5, 7, 6, 7, 8, 8, 8, 8, 6,
    7, 6, 7, 8, 8, 8, 8, 1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7,
    6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 3, 4, 3, 4, 7, 7, 7, 7, 3, 4, 3,
    4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 3, 4, 3, 4,
    7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8,
    8, 8, 8, 1, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6, 7, 6, 7, 6, 7, 8, 8,
    8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 2, 3, 2, 3, 6, 7, 6, 7, 2, 3, 2, 3, 6, 7, 6,
    7, 6, 7, 6, 7, 8, 8, 8, 8, 6, 7, 6, 7, 8, 8, 8, 8, 3, 4, 3, 4, 7, 7, 7, 7,
    3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7, 7, 7, 8, 8, 8, 8, 3,
    4, 3, 4, 7, 7, 7, 7, 3, 4, 3, 4, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 7, 7,
    7, 7, 8, 8, 8, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HL_LOOKUP: [u8; 256] = [
    0, 5, 1, 6, 3, 7, 3, 7, 1, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7, 4, 7, 3,
    7, 3, 7, 4, 7, 4, 7, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7,
    3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 5, 8, 6, 8, 7, 8, 7, 8, 6, 8, 6,
    8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6, 8, 6, 8,
    7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7,
    8, 7, 8, 1, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3, 7, 3, 7, 3, 7, 4, 7,
    4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 2, 6, 2, 6, 3, 7, 3, 7, 2, 6, 2, 6, 3, 7, 3,
    7, 3, 7, 3, 7, 4, 7, 4, 7, 3, 7, 3, 7, 4, 7, 4, 7, 6, 8, 6, 8, 7, 8, 7, 8,
    6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 6,
    8, 6, 8, 7, 8, 7, 8, 6, 8, 6, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8, 7, 8,
    7, 8, 7, 8, 7, 8,
];

#[rustfmt::skip]
const ZERO_CTX_HH_LOOKUP: [u8; 256] = [
    0, 1, 3, 4, 1, 2, 4, 5, 3, 4, 6, 7, 4, 5, 7, 7, 1, 2, 4, 5, 2, 2, 5, 5, 4,
    5, 7, 7, 5, 5, 7, 7, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5,
    7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 1, 2, 4, 5, 2, 2, 5, 5, 4, 5, 7,
    7, 5, 5, 7, 7, 2, 2, 5, 5, 2, 2, 5, 5, 5, 5, 7, 7, 5, 5, 7, 7, 4, 5, 7, 7,
    5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7,
    7, 8, 8, 3, 4, 6, 7, 4, 5, 7, 7, 6, 7, 8, 8, 7, 7, 8, 8, 4, 5, 7, 7, 5, 5,
    7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 6, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8,
    8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 4, 5, 7, 7, 5, 5, 7, 7,
    7, 7, 8, 8, 7, 7, 8, 8, 5, 5, 7, 7, 5, 5, 7, 7, 7, 7, 8, 8, 7, 7, 8, 8, 7,
    7, 8, 8, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 7, 7, 8, 8, 7, 7, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8,
];

const FLAG_SIGNIFICANT: u8 = 1 << 0;
const FLAG_ZERO_CODED: u8 = 1 << 1;
const FLAG_REFINED: u8 = 1 << 2;

struct CodeBlockScratch {
    states: Vec<u8>,
    neighbors: Vec<u8>,
    signs: Vec<u8>,
    mags: Vec<u32>,
    contexts: [MqContext; 19],
}

impl CodeBlockScratch {
    fn new() -> Self {
        Self {
            states: Vec::with_capacity(66 * 66),
            neighbors: Vec::with_capacity(66 * 66),
            signs: Vec::with_capacity(66 * 66),
            mags: Vec::with_capacity(64 * 64),
            contexts: [MqContext::default(); 19],
        }
    }

    fn reset_contexts(&mut self) {
        for ctx in &mut self.contexts {
            *ctx = MqContext::default();
        }
        self.contexts[0].reset_with_index(4);
        self.contexts[17].reset_with_index(3);
        self.contexts[18].reset_with_index(46);
    }

    #[inline(always)]
    fn idx(padded_w: usize, x: u32, y: u32) -> usize {
        (y as usize + 1) * padded_w + (x as usize + 1)
    }

    #[inline(always)]
    fn mark_significant(&mut self, padded_w: usize, x: u32, y: u32) {
        let i = Self::idx(padded_w, x, y);
        self.states[i] |= FLAG_SIGNIFICANT;

        // Neighbor significance bit order (MSB -> LSB) matches `NeighborSignificances`:
        // 7: top-left, 6: top, 5: top-right, 4: left, 3: bottom-left, 2: right, 1: bottom-right, 0: bottom.
        self.neighbors[i - padded_w - 1] |= 1 << 1; // top-left's bottom-right
        self.neighbors[i - padded_w] |= 1 << 0; // top's bottom
        self.neighbors[i - padded_w + 1] |= 1 << 3; // top-right's bottom-left
        self.neighbors[i - 1] |= 1 << 2; // left's right
        self.neighbors[i + 1] |= 1 << 4; // right's left
        self.neighbors[i + padded_w - 1] |= 1 << 5; // bottom-left's top-right
        self.neighbors[i + padded_w] |= 1 << 6; // bottom's top
        self.neighbors[i + padded_w + 1] |= 1 << 7; // bottom-right's top-left
    }

    #[inline(always)]
    fn zero_ctx_label(&self, kind: SubBandKind, neighbors: u8) -> u8 {
        match kind {
            SubBandKind::LowLow | SubBandKind::LowHigh => {
                ZERO_CTX_LL_LH_LOOKUP[neighbors as usize]
            }
            SubBandKind::HighLow => ZERO_CTX_HL_LOOKUP[neighbors as usize],
            SubBandKind::HighHigh => ZERO_CTX_HH_LOOKUP[neighbors as usize],
        }
    }

    #[inline(always)]
    fn encode_sign(&mut self, mq: &mut MqEncoder, padded_w: usize, x: u32, y: u32) {
        let i = Self::idx(padded_w, x, y);
        let sigs = self.neighbors[i] & 0b0101_0101;
        let top_sign = self.signs[i - padded_w];
        let left_sign = self.signs[i - 1];
        let right_sign = self.signs[i + 1];
        let bottom_sign = self.signs[i + padded_w];

        let packed_signs = (top_sign << 6) | (left_sign << 4) | (right_sign << 2) | bottom_sign;
        let neg_sigs = sigs & packed_signs;
        let pos_sigs = sigs & !packed_signs;
        let merged = (neg_sigs << 1) | pos_sigs;
        let (ctx_label, xor_bit) = SIGN_CONTEXT_LOOKUP[merged as usize];
        let coeff_sign = self.signs[i] as u32;
        mq.encode_bit(
            &mut self.contexts[ctx_label as usize],
            coeff_sign ^ (xor_bit as u32),
        );
    }
}

struct EncodedCodeBlock {
    included: bool,
    missing_bitplanes: u8,
    coding_passes: u8,
    bytes: Vec<u8>,
}

fn encode_code_block(
    sb: &QuantizedSubBand,
    x0: u32,
    y0: u32,
    cb_w: u32,
    cb_h: u32,
    scratch: &mut CodeBlockScratch,
) -> EncodedCodeBlock {
    let padded_w = (cb_w as usize) + 2;
    let total_padded = padded_w * ((cb_h as usize) + 2);
    scratch.states.clear();
    scratch.states.resize(total_padded, 0);
    scratch.neighbors.clear();
    scratch.neighbors.resize(total_padded, 0);
    scratch.signs.clear();
    scratch.signs.resize(total_padded, 0);
    scratch.mags.clear();

    let mut max_mag = 0u32;
    let sb_w = sb.width as usize;
    for dy in 0..cb_h {
        let sy = (y0 + dy) as usize;
        for dx in 0..cb_w {
            let s_idx = sy * sb_w + (x0 + dx) as usize;
            let mag = sb.mags[s_idx];
            if mag > max_mag {
                max_mag = mag;
            }
            scratch.mags.push(mag);
            scratch.signs[CodeBlockScratch::idx(padded_w, dx, dy)] = sb.signs[s_idx];
        }
    }

    if max_mag == 0 || sb.num_bitplanes == 0 {
        return EncodedCodeBlock {
            included: false,
            missing_bitplanes: 0,
            coding_passes: 0,
            bytes: Vec::new(),
        };
    }

    let highest_bp = 31 - max_mag.leading_zeros() as u8;
    let actual_bitplanes = (highest_bp + 1).min(sb.num_bitplanes);
    let missing_bitplanes = sb.num_bitplanes - actual_bitplanes;
    let coding_passes = 1 + 3 * (actual_bitplanes - 1);

    scratch.reset_contexts();
    let mut mq = MqEncoder::new();
    let kind = sb.kind;

    let encode_cleanup_bitplane = |bp: u8, scratch: &mut CodeBlockScratch, mq: &mut MqEncoder| {
        for base_row in (0..cb_h).step_by(4) {
            let stripe_end = (base_row + 4).min(cb_h);
            let stripe_len = stripe_end - base_row;
            for x in 0..cb_w {
                let mut dy = base_row;
                if stripe_len == 4 {
                    let i0 = CodeBlockScratch::idx(padded_w, x, dy);
                    let i1 = i0 + padded_w;
                    let i2 = i1 + padded_w;
                    let i3 = i2 + padded_w;
                    let all_clean = (scratch.states[i0] & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED))
                        == 0
                        && (scratch.states[i1] & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED)) == 0
                        && (scratch.states[i2] & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED)) == 0
                        && (scratch.states[i3] & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED)) == 0;
                    let all_zero_ctx = scratch.neighbors[i0] == 0
                        && scratch.neighbors[i1] == 0
                        && scratch.neighbors[i2] == 0
                        && scratch.neighbors[i3] == 0;
                    if all_clean && all_zero_ctx {
                        let m0 = (dy as usize) * (cb_w as usize) + (x as usize);
                        let b0 = (scratch.mags[m0] >> bp) & 1;
                        let b1 = (scratch.mags[m0 + cb_w as usize] >> bp) & 1;
                        let b2 = (scratch.mags[m0 + 2 * (cb_w as usize)] >> bp) & 1;
                        let b3 = (scratch.mags[m0 + 3 * (cb_w as usize)] >> bp) & 1;
                        let first_sig = if b0 != 0 {
                            Some(0u32)
                        } else if b1 != 0 {
                            Some(1)
                        } else if b2 != 0 {
                            Some(2)
                        } else if b3 != 0 {
                            Some(3)
                        } else {
                            None
                        };
                        match first_sig {
                            None => {
                                mq.encode_bit(&mut scratch.contexts[17], 0);
                                continue;
                            }
                            Some(k) => {
                                mq.encode_bit(&mut scratch.contexts[17], 1);
                                mq.encode_bit(&mut scratch.contexts[18], (k >> 1) & 1);
                                mq.encode_bit(&mut scratch.contexts[18], k & 1);
                                dy += k;
                                scratch.encode_sign(mq, padded_w, x, dy);
                                scratch.mark_significant(padded_w, x, dy);
                                dy += 1;
                            }
                        }
                    }
                }

                while dy < stripe_end {
                    let i = CodeBlockScratch::idx(padded_w, x, dy);
                    if (scratch.states[i] & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED)) == 0 {
                        let label = scratch.zero_ctx_label(kind, scratch.neighbors[i]);
                        let bit = (scratch.mags[(dy as usize) * (cb_w as usize) + (x as usize)]
                            >> bp)
                            & 1;
                        mq.encode_bit(&mut scratch.contexts[label as usize], bit);
                        if bit == 1 {
                            scratch.encode_sign(mq, padded_w, x, dy);
                            scratch.mark_significant(padded_w, x, dy);
                        }
                    }
                    dy += 1;
                }
            }
        }
        for s in &mut scratch.states {
            *s &= !FLAG_ZERO_CODED;
        }
    };

    // Most significant active bitplane: Cleanup pass only.
    encode_cleanup_bitplane(actual_bitplanes - 1, scratch, &mut mq);

    for bp in (0..(actual_bitplanes - 1)).rev() {
        // 1. Significance Propagation Pass (SPP)
        for base_row in (0..cb_h).step_by(4) {
            let stripe_end = (base_row + 4).min(cb_h);
            for x in 0..cb_w {
                for dy in base_row..stripe_end {
                    let i = CodeBlockScratch::idx(padded_w, x, dy);
                    if (scratch.states[i] & FLAG_SIGNIFICANT) == 0 && scratch.neighbors[i] != 0 {
                        let label = scratch.zero_ctx_label(kind, scratch.neighbors[i]);
                        let bit = (scratch.mags[(dy as usize) * (cb_w as usize) + (x as usize)]
                            >> bp)
                            & 1;
                        mq.encode_bit(&mut scratch.contexts[label as usize], bit);
                        scratch.states[i] |= FLAG_ZERO_CODED;
                        if bit == 1 {
                            scratch.encode_sign(&mut mq, padded_w, x, dy);
                            scratch.mark_significant(padded_w, x, dy);
                        }
                    }
                }
            }
        }

        // 2. Magnitude Refinement Pass (MRP)
        for base_row in (0..cb_h).step_by(4) {
            let stripe_end = (base_row + 4).min(cb_h);
            for x in 0..cb_w {
                for dy in base_row..stripe_end {
                    let i = CodeBlockScratch::idx(padded_w, x, dy);
                    let st = scratch.states[i];
                    if (st & (FLAG_SIGNIFICANT | FLAG_ZERO_CODED)) == FLAG_SIGNIFICANT {
                        let ctx_label = if (st & FLAG_REFINED) != 0 {
                            16
                        } else if scratch.neighbors[i] != 0 {
                            15
                        } else {
                            14
                        };
                        let bit = (scratch.mags[(dy as usize) * (cb_w as usize) + (x as usize)]
                            >> bp)
                            & 1;
                        mq.encode_bit(&mut scratch.contexts[ctx_label], bit);
                        scratch.states[i] |= FLAG_REFINED;
                    }
                }
            }
        }

        // 3. Cleanup Pass (CP)
        encode_cleanup_bitplane(bp, scratch, &mut mq);
    }

    let bytes = mq.flush();
    EncodedCodeBlock {
        included: !bytes.is_empty(),
        missing_bitplanes,
        coding_passes,
        bytes,
    }
}

/// Bit-level writer enforcing ITU-T T.800 Annex B.10 packet header 0xFF byte stuffing.
struct PacketBitWriter {
    bytes: Vec<u8>,
    cur: u8,
    bits_in_cur: u8,
}

impl PacketBitWriter {
    fn new() -> Self {
        Self {
            bytes: Vec::with_capacity(64),
            cur: 0,
            bits_in_cur: 0,
        }
    }

    #[inline(always)]
    fn write_bit(&mut self, bit: u32) {
        self.cur = (self.cur << 1) | ((bit & 1) as u8);
        self.bits_in_cur += 1;
        if self.bits_in_cur == 8 {
            let finished = self.cur;
            self.bytes.push(finished);
            self.cur = 0;
            // After writing 0xFF, next byte starts with a forced 0 stuff bit in MSB.
            self.bits_in_cur = if finished == 0xFF { 1 } else { 0 };
        }
    }

    fn write_bits(&mut self, val: u32, count: u8) {
        for shift in (0..count).rev() {
            self.write_bit((val >> shift) & 1);
        }
    }

    fn finish_aligned(mut self) -> Vec<u8> {
        if self.bits_in_cur > 0 {
            let shifted = self.cur << (8 - self.bits_in_cur);
            self.bytes.push(shifted);
            if shifted == 0xFF {
                self.bytes.push(0);
            }
        }
        self.bytes
    }
}

/// Tag-tree encoder matching `hayro-jpeg2000` `TagNode` quadtree semantics (ITU-T T.800 B.10.2).
struct TagTreeNodeEnc {
    top_left_w: u32,
    top_left_h: u32,
    level: u16,
    value: u32,
    min_descendant: u32,
    initialized: bool,
    children: [usize; 4],
}

struct TagTreeEncoder {
    root: usize,
    nodes: Vec<TagTreeNodeEnc>,
}

impl TagTreeEncoder {
    fn new(width: u32, height: u32, leaf_values: &[u32]) -> Self {
        let mut nodes = Vec::new();
        let root_level = if width == 0 || height == 0 {
            0
        } else {
            width
                .next_power_of_two()
                .ilog2()
                .max(height.next_power_of_two().ilog2()) as u16
        };

        #[allow(clippy::too_many_arguments)]
        fn build(
            w: u32,
            h: u32,
            x0: u32,
            y0: u32,
            total_w: u32,
            level: u16,
            leaf_values: &[u32],
            nodes: &mut Vec<TagTreeNodeEnc>,
        ) -> usize {
            if w == 0 || h == 0 {
                return usize::MAX;
            }
            if level == 0 {
                let val = leaf_values[(y0 as usize) * (total_w as usize) + (x0 as usize)];
                let idx = nodes.len();
                nodes.push(TagTreeNodeEnc {
                    top_left_w: w,
                    top_left_h: h,
                    level: 0,
                    value: 0,
                    min_descendant: val,
                    initialized: false,
                    children: [usize::MAX; 4],
                });
                return idx;
            }

            let half = 1u32 << (level - 1);
            let tlw = half.min(w);
            let tlh = half.min(h);
            let c0 = build(tlw, tlh, x0, y0, total_w, level - 1, leaf_values, nodes);
            let c1 = build(
                w - tlw,
                tlh,
                x0 + tlw,
                y0,
                total_w,
                level - 1,
                leaf_values,
                nodes,
            );
            let c2 = build(
                tlw,
                h - tlh,
                x0,
                y0 + tlh,
                total_w,
                level - 1,
                leaf_values,
                nodes,
            );
            let c3 = build(
                w - tlw,
                h - tlh,
                x0 + tlw,
                y0 + tlh,
                total_w,
                level - 1,
                leaf_values,
                nodes,
            );
            let mut min_descendant = u32::MAX;
            for c in [c0, c1, c2, c3] {
                if c != usize::MAX {
                    min_descendant = min_descendant.min(nodes[c].min_descendant);
                }
            }
            let idx = nodes.len();
            nodes.push(TagTreeNodeEnc {
                top_left_w: tlw,
                top_left_h: tlh,
                level,
                value: 0,
                min_descendant,
                initialized: false,
                children: [c0, c1, c2, c3],
            });
            idx
        }

        let root = build(
            width,
            height,
            0,
            0,
            width,
            root_level,
            leaf_values,
            &mut nodes,
        );
        Self { root, nodes }
    }

    fn write(&mut self, x: u32, y: u32, max_val: u32, writer: &mut PacketBitWriter) {
        if self.root == usize::MAX {
            return;
        }
        fn step(
            idx: usize,
            x: u32,
            y: u32,
            parent_val: u32,
            max_val: u32,
            nodes: &mut [TagTreeNodeEnc],
            writer: &mut PacketBitWriter,
        ) {
            let node = &mut nodes[idx];
            if !node.initialized {
                let mut val = node.value.max(parent_val);
                let target = node.min_descendant;
                while val < max_val {
                    if val < target {
                        writer.write_bit(0);
                        val += 1;
                    } else {
                        writer.write_bit(1);
                        node.initialized = true;
                        break;
                    }
                }
                node.value = val;
            }
            if node.value >= max_val || node.level == 0 {
                return;
            }
            let tlw = node.top_left_w;
            let tlh = node.top_left_h;
            let node_val = node.value;
            let children = node.children;
            match (x < tlw, y < tlh) {
                (true, true) => step(children[0], x, y, node_val, max_val, nodes, writer),
                (false, true) => step(children[1], x - tlw, y, node_val, max_val, nodes, writer),
                (true, false) => step(children[2], x, y - tlh, node_val, max_val, nodes, writer),
                (false, false) => step(
                    children[3],
                    x - tlw,
                    y - tlh,
                    node_val,
                    max_val,
                    nodes,
                    writer,
                ),
            }
        }
        step(self.root, x, y, 0, max_val, &mut self.nodes, writer);
    }
}

fn write_coding_passes(writer: &mut PacketBitWriter, passes: u8) {
    match passes {
        1 => writer.write_bit(0),
        2 => writer.write_bits(0b10, 2),
        3 => writer.write_bits(0b1100, 4),
        4 => writer.write_bits(0b1101, 4),
        5 => writer.write_bits(0b1110, 4),
        6..=36 => {
            writer.write_bits(0b1111, 4);
            writer.write_bits(u32::from(passes - 6), 5);
        }
        37..=164 => {
            writer.write_bits(0x1FF, 9);
            writer.write_bits(u32::from(passes - 37), 7);
        }
        _ => unreachable!("coding passes bounded by 30 bitplanes (<= 88 passes)"),
    }
}

/// Computes codeblock grid dimensions matching `build.rs::build_precincts` for precinct at `(0,0)`
/// covering a single subband starting at `(0,0)` with dimensions `(w, h)`.
struct EncodedSubBand {
    cb_x: u32,
    cb_y: u32,
    blocks: Vec<EncodedCodeBlock>,
}

fn encode_subband(sb: &QuantizedSubBand, scratch: &mut CodeBlockScratch) -> EncodedSubBand {
    if sb.width == 0 || sb.height == 0 {
        return EncodedSubBand {
            cb_x: 0,
            cb_y: 0,
            blocks: Vec::new(),
        };
    }
    let cb_x = sb.width.div_ceil(CB_DIM);
    let cb_y = sb.height.div_ceil(CB_DIM);
    let mut blocks = Vec::with_capacity((cb_x * cb_y) as usize);
    for by in 0..cb_y {
        let y0 = by * CB_DIM;
        let bh = (sb.height - y0).min(CB_DIM);
        for bx in 0..cb_x {
            let x0 = bx * CB_DIM;
            let bw = (sb.width - x0).min(CB_DIM);
            blocks.push(encode_code_block(sb, x0, y0, bw, bh, scratch));
        }
    }
    EncodedSubBand { cb_x, cb_y, blocks }
}

fn write_packet(subbands: &[&EncodedSubBand], out_tile_data: &mut Vec<u8>) {
    let any_included = subbands
        .iter()
        .any(|sb| sb.blocks.iter().any(|b| b.included));

    if !any_included {
        out_tile_data.push(0x00);
        return;
    }

    let mut hdr = PacketBitWriter::new();
    hdr.write_bit(1);

    for sb in subbands {
        if sb.blocks.is_empty() {
            continue;
        }
        let inc_vals: Vec<u32> = sb
            .blocks
            .iter()
            .map(|b| if b.included { 0 } else { 1 })
            .collect();
        let zero_vals: Vec<u32> = sb
            .blocks
            .iter()
            .map(|b| u32::from(b.missing_bitplanes))
            .collect();
        let mut inc_tree = TagTreeEncoder::new(sb.cb_x, sb.cb_y, &inc_vals);
        let mut zb_tree = TagTreeEncoder::new(sb.cb_x, sb.cb_y, &zero_vals);

        for by in 0..sb.cb_y {
            for bx in 0..sb.cb_x {
                let idx = (by * sb.cb_x + bx) as usize;
                let cb = &sb.blocks[idx];
                inc_tree.write(bx, by, 1, &mut hdr);
                if !cb.included {
                    continue;
                }
                zb_tree.write(bx, by, u32::MAX, &mut hdr);
                write_coding_passes(&mut hdr, cb.coding_passes);

                let data_len = cb.bytes.len();
                let ilog_passes = cb.coding_passes.ilog2();
                let mut l_block = 3u32;
                let mut k = 0u32;
                while (1usize << (l_block + ilog_passes)) <= data_len {
                    l_block += 1;
                    k += 1;
                }
                for _ in 0..k {
                    hdr.write_bit(1);
                }
                hdr.write_bit(0);
                let len_bits = (l_block + ilog_passes) as u8;
                hdr.write_bits(data_len as u32, len_bits);
            }
        }
    }

    out_tile_data.extend_from_slice(&hdr.finish_aligned());
    for sb in subbands {
        for cb in &sb.blocks {
            if cb.included {
                out_tile_data.extend_from_slice(&cb.bytes);
            }
        }
    }
}

fn write_qcd_payload(out: &mut Vec<u8>, guard_bits: u8, steps: &[QuantStepParams]) {
    let sqcd = (guard_bits << 5) | 2; // ScalarExpounded (style = 2)
    out.push(sqcd);
    for step in steps {
        let word: u16 = ((step.exponent & 0x1F) << 11) | (step.mantissa & 0x07FF);
        out.extend_from_slice(&word.to_be_bytes());
    }
}

fn encode_jpx_components(
    components: SmallVec<[Vec<f32>; 3]>,
    width: u32,
    height: u32,
    quality: u8,
    use_mct: bool,
    bit_depth: u8,
) -> Result<Vec<u8>, String> {
    let num_comps = components.len();
    // Single precinct per resolution requires width/height <= 16384; clamp decompositions safely.
    let n_l = (width.min(height).max(1).ilog2() as u8).min(MAX_DECOMP_LEVELS);
    let depth_scale = 2.0_f32.powi((bit_depth as i32) - (BIT_DEPTH as i32));
    let base_step = if bit_depth == 1 {
        0.08
    } else {
        quality_to_base_step(quality) * depth_scale
    };

    let mut comp_qsteps: SmallVec<[_; 3]> = SmallVec::with_capacity(num_comps);
    let mut comp_encoded_sbs: SmallVec<[_; 3]> = SmallVec::with_capacity(num_comps);
    let mut scratch = CodeBlockScratch::new();

    for (comp_idx, comp_data) in components.into_iter().enumerate() {
        let comp_step = if use_mct && comp_idx > 0 {
            base_step * CHROMA_STEP_MULT
        } else {
            base_step
        };
        let dwt_sbs = forward_dwt_2d(comp_data, width, height, n_l);
        let (guard_bits, qsteps, q_sbs) =
            quantize_component(&dwt_sbs, n_l, comp_step, bit_depth);
        let enc_sbs: Vec<EncodedSubBand> = q_sbs
            .iter()
            .map(|qsb| encode_subband(qsb, &mut scratch))
            .collect();
        comp_qsteps.push((guard_bits, qsteps));
        comp_encoded_sbs.push(enc_sbs);
    }

    // Emit tile body packets in LRCP order: resolution 0..=N_L, component 0..num_comps.
    let mut tile_packets = Vec::with_capacity((width * height) as usize / 8 + 256);
    for res in 0..=n_l {
        for comp_sbs in &comp_encoded_sbs {
            if res == 0 {
                write_packet(&[&comp_sbs[0]], &mut tile_packets);
            } else {
                let base_idx = 1 + ((res - 1) as usize) * 3;
                write_packet(
                    &[
                        &comp_sbs[base_idx],
                        &comp_sbs[base_idx + 1],
                        &comp_sbs[base_idx + 2],
                    ],
                    &mut tile_packets,
                );
            }
        }
    }

    let num_subbands = 1 + 3 * (n_l as usize);
    let mut codestream = Vec::with_capacity(256 + tile_packets.len());

    // SOC
    codestream.extend_from_slice(&[0xFF, 0x4F]);

    // SIZ (Annex A.5.1)
    let l_siz = (38 + 3 * num_comps) as u16;
    codestream.extend_from_slice(&[0xFF, 0x51]);
    codestream.extend_from_slice(&l_siz.to_be_bytes());
    codestream.extend_from_slice(&0u16.to_be_bytes()); // Rsiz = baseline Part 1
    codestream.extend_from_slice(&width.to_be_bytes()); // Xsiz
    codestream.extend_from_slice(&height.to_be_bytes()); // Ysiz
    codestream.extend_from_slice(&0u32.to_be_bytes()); // XOsiz
    codestream.extend_from_slice(&0u32.to_be_bytes()); // YOsiz
    codestream.extend_from_slice(&width.to_be_bytes()); // XTsiz
    codestream.extend_from_slice(&height.to_be_bytes()); // YTsiz
    codestream.extend_from_slice(&0u32.to_be_bytes()); // XTOsiz
    codestream.extend_from_slice(&0u32.to_be_bytes()); // YTOsiz
    codestream.extend_from_slice(&(num_comps as u16).to_be_bytes()); // Csiz
    for _ in 0..num_comps {
        codestream.extend_from_slice(&[bit_depth.saturating_sub(1), 1, 1]); // unsigned bit_depth, XRsiz=1, YRsiz=1
    }

    // COD (Annex A.6.1)
    codestream.extend_from_slice(&[0xFF, 0x52]);
    codestream.extend_from_slice(&12u16.to_be_bytes());
    codestream.push(0x00); // Scod: default precincts (PPx=PPy=15), no SOP/EPH
    codestream.push(0x00); // SGcod progression: LRCP
    codestream.extend_from_slice(&1u16.to_be_bytes()); // SGcod layers = 1
    codestream.push(if use_mct { 0x01 } else { 0x00 }); // MCT
    codestream.push(n_l); // decomposition levels
    codestream.push(4); // xcb - 2 -> 2^6 = 64
    codestream.push(4); // ycb - 2 -> 2^6 = 64
    codestream.push(0x00); // code-block style: default MQ
    codestream.push(0x00); // wavelet transform: 9-7 irreversible

    // QCD (component 0 / default)
    let l_qcd = (3 + 2 * num_subbands) as u16;
    codestream.extend_from_slice(&[0xFF, 0x5C]);
    codestream.extend_from_slice(&l_qcd.to_be_bytes());
    write_qcd_payload(&mut codestream, comp_qsteps[0].0, &comp_qsteps[0].1);

    // QCC for chroma components (1, 2) if MCT enabled with custom chroma quantization
    if use_mct {
        for (comp_idx, (guard, qsteps)) in comp_qsteps.iter().enumerate().skip(1) {
            let l_qcc = (4 + 2 * num_subbands) as u16;
            codestream.extend_from_slice(&[0xFF, 0x5D]);
            codestream.extend_from_slice(&l_qcc.to_be_bytes());
            codestream.push(comp_idx as u8);
            write_qcd_payload(&mut codestream, *guard, qsteps);
        }
    }

    // SOT + SOD + packets + EOC
    let psot = (14 + tile_packets.len()) as u32; // 12 (SOT) + 2 (SOD) + packets
    codestream.extend_from_slice(&[0xFF, 0x90]);
    codestream.extend_from_slice(&10u16.to_be_bytes());
    codestream.extend_from_slice(&0u16.to_be_bytes()); // Isot
    codestream.extend_from_slice(&psot.to_be_bytes());
    codestream.push(0); // TPsot
    codestream.push(1); // TNsot
    codestream.extend_from_slice(&[0xFF, 0x93]); // SOD
    codestream.extend_from_slice(&tile_packets);
    codestream.extend_from_slice(&[0xFF, 0xD9]); // EOC

    Ok(codestream)
}

#[cfg(test)]
mod tests {
    use super::*;
    use hayro_jpeg2000::{DecodeSettings, Image as JpxImage};
    use image::DynamicImage;

    fn decode_strict_j2k(bytes: &[u8]) -> DynamicImage {
        let settings = DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        };
        let img = JpxImage::new(bytes, &settings).expect("hayro parse j2k header strictly");
        DynamicImage::from_decoder(img).expect("hayro strict j2k decode")
    }

    fn compute_psnr(a: &[u8], b: &[u8]) -> f64 {
        assert_eq!(a.len(), b.len());
        let mse: f64 = a
            .iter()
            .zip(b.iter())
            .map(|(&x, &y)| {
                let d = (x as f64) - (y as f64);
                d * d
            })
            .sum::<f64>()
            / (a.len() as f64);
        if mse <= 1e-12 {
            99.0
        } else {
            10.0 * ((255.0 * 255.0) / mse).log10()
        }
    }

    #[test]
    fn roundtrips_grayscale_and_edge_dimensions_under_strict_hayro_decode() {
        for &(w, h) in &[(1u32, 1u32), (7, 5), (65, 129), (128, 128)] {
            let mut gray = vec![0u8; (w * h) as usize];
            for y in 0..h {
                for x in 0..w {
                    gray[(y * w + x) as usize] =
                        (((x * 13 + y * 29) & 0xFF) ^ ((x ^ y) * 7) & 0x7F) as u8;
                }
            }
            let encoded = encode_jpx_gray(&gray, w, h, 95).expect("encode gray j2k");
            let decoded = decode_strict_j2k(&encoded);
            assert_eq!((decoded.width(), decoded.height()), (w, h));
            let dec_gray = decoded.into_luma8().into_raw();
            let psnr = compute_psnr(&gray, &dec_gray);
            assert!(
                psnr >= 30.0,
                "grayscale Q95 {w}x{h} PSNR unexpectedly low: {psnr:.2} dB"
            );
        }
    }

    #[test]
    fn roundtrips_rgb_with_mct_high_psnr_and_monotone_quality_sizes() {
        let w = 160u32;
        let h = 120u32;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let idx = ((y * w + x) * 3) as usize;
                // Smooth realistic photo-like gradient + localized pattern
                let fx = x as f64 / w as f64;
                let fy = y as f64 / h as f64;
                rgb[idx] = (40.0 + 180.0 * fx + 25.0 * (fy * 12.0).sin()) as u8;
                rgb[idx + 1] = (30.0 + 160.0 * fy + 30.0 * (fx * 10.0).cos()) as u8;
                rgb[idx + 2] = (200.0 - 140.0 * fx * fy) as u8;
            }
        }

        let j2k_q20 = encode_jpx_rgb(&rgb, w, h, 20).expect("j2k q20");
        let j2k_q50 = encode_jpx_rgb(&rgb, w, h, 50).expect("j2k q50");
        let j2k_q80 = encode_jpx_rgb(&rgb, w, h, 80).expect("j2k q80");
        let j2k_q100 = encode_jpx_rgb(&rgb, w, h, 100).expect("j2k q100");

        assert!(j2k_q20.len() < j2k_q50.len());
        assert!(j2k_q50.len() < j2k_q80.len());
        assert!(j2k_q80.len() < j2k_q100.len());

        for (q, bytes, min_psnr) in [
            (20, &j2k_q20, 26.0),
            (50, &j2k_q50, 32.0),
            (80, &j2k_q80, 37.0),
            (100, &j2k_q100, 42.0),
        ] {
            let dec = decode_strict_j2k(bytes);
            assert_eq!((dec.width(), dec.height()), (w, h));
            let psnr = compute_psnr(&rgb, &dec.into_rgb8().into_raw());
            assert!(
                psnr >= min_psnr,
                "RGB Q{q} PSNR too low: {psnr:.2} dB (bytes={})",
                bytes.len()
            );
        }
    }

    #[test]
    fn mq_encoder_handles_carry_and_0xff_stuffing_on_noise() {
        let w = 97u32;
        let h = 99u32;
        let mut seed = 0xDEAD_BEEF_1234_5678u64;
        let mut rgb = vec![0u8; (w * h * 3) as usize];
        for b in &mut rgb {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
            *b = (seed >> 56) as u8;
        }
        let encoded = encode_jpx_rgb(&rgb, w, h, 75).expect("encode noise j2k");
        let dec = decode_strict_j2k(&encoded);
        assert_eq!((dec.width(), dec.height()), (w, h));
    }

    #[test]
    fn roundtrips_1bit_bilevel_and_10bit_hdr_under_strict_hayro_decode() {
        let w = 64u32;
        let h = 64u32;
        let mut bw = vec![0u8; (w * h) as usize];
        for y in 0..h {
            for x in 0..w {
                bw[(y * w + x) as usize] = if ((x / 8) + (y / 8)) % 2 == 0 {
                    255
                } else {
                    0
                };
            }
        }
        let j2k_1bit = encode_jpx_bilevel_1bit(&bw, w, h, 85).expect("1-bit j2k");
        let dec_1bit = decode_strict_j2k(&j2k_1bit);
        assert_eq!((dec_1bit.width(), dec_1bit.height()), (w, h));

        let mut hdr_rgb = vec![0u8; (w * h * 3) as usize];
        for y in 0..h {
            for x in 0..w {
                let p = ((y * w + x) * 3) as usize;
                hdr_rgb[p] = ((x * 255) / (w - 1)) as u8;
                hdr_rgb[p + 1] = ((y * 255) / (h - 1)) as u8;
                hdr_rgb[p + 2] = 160;
            }
        }
        let j2k_10bit = encode_jpx_rgb_10bit(&hdr_rgb, w, h, 85).expect("10-bit hdr j2k");
        let jpx_hdr = hayro_jpeg2000::Image::new(&j2k_10bit, &hayro_jpeg2000::DecodeSettings {
            resolve_palette_indices: true,
            strict: true,
            target_resolution: None,
        })
        .expect("parse 10-bit hdr j2k");
        assert_eq!(jpx_hdr.original_bit_depth(), 10);
        let dec_10bit = DynamicImage::from_decoder(jpx_hdr).expect("decode 10-bit hdr j2k");
        assert_eq!((dec_10bit.width(), dec_10bit.height()), (w, h));
    }
}
