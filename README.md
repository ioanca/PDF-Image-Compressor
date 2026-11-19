# PDF Image Compressor

A fast, lightweight, single-executable desktop GUI application written in Rust for compressing PDF files containing scanned pages, documents, or photos while preserving physical page dimensions (`/MediaBox`, `/CropBox`) and visual layout (`cm` transformation matrices).

Built with `eframe`/`egui` (`glow` OpenGL backend), `lopdf`, `hayro`, `mozjpeg`, `flate2` (`zlib-rs`), `rayon`, and `mimalloc`, it combines a unified single- and multi-file explorer interface, real-time empirical size estimation, automatic target-size solving, interactive full-page visual diff previews, and low-RAM structural PDF optimization.

---

## Key Features

### Unified Single-File & Batch Interface
- **Explorer-Style File Table:** Load one or more unencrypted `.pdf` files via **Select** (`rfd`) or drag-and-drop. Both single-file and multi-file sessions share the same unified interface with multi-row selection (`Click`, `Ctrl/Cmd+Click`, `Shift+Click`, `Ctrl/Cmd+A`), **Apply for ALL** (visible when `> 1` file is loaded, keeping **Show Preview** pinned at the exact same vertical position), **Show Preview**, and **Compress**.
- **Clickable Estimation Column:** Click the `Est. (MB)` / `Est. (%)` table header to toggle between estimated output size in megabytes and percentage of original size.
- **DPI-Aware Responsive Layout:** Main control panel locks to `27.5%` of logical monitor width (`42.5%` default preview pane when expanded), capping the top file list at 10 visible rows, hiding **Compression Summary** when no file is selected, scaling the summary table dynamically between 5 and 10 visible rows when `≥ 5` files are loaded, and enforcing a dynamic `MinInnerSize` matched to content height.

### Compression & Resize Settings
- **Resize Filter:**
  - `Fancy` (`Lanczos3`) for sharper downsampling
  - `Fast` (`CatmullRom`) for faster resizing
- **Image Format (PDF-Native Output Codecs):**
  - **JPG (`/DCTDecode`):** Powered by MozJPEG (`mozjpeg`) with trellis quantization, progressive scan optimization, overshoot deringing, and adaptive `4:2:0` (`quality <= 85`) / `4:4:4` (`quality > 85`) chroma subsampling.
  - **JPEG 2000 (`/JPXDecode`):** Pure-Rust ITU-T T.800 Part 1 encoder (`jpx_enc`) using Irreversible Color Transform (ICT), CDF 9/7 DWT, and EBCOT arithmetic coding.
  - **PNG (`/FlateDecode`):** Lossless encoding using adaptive PNG row filtering (PDF `/Predictor 15`) with raw zlib fallback and 1-bit bi-level packing.
- **Format-Adaptive Color Depth:** Automatically filters available options to those supported by the active format (falling back cleanly when switching codecs):
  - `Black/White (1-bit)` — `JPEG 2000`, `PNG`
  - `Grayscale (8-bit)` — `JPG`, `JPEG 2000`, `PNG`
  - `High Color (16-bit)` — `JPEG 2000`, `PNG` (RGB565 quantization)
  - `True Color (24-bit)` — `JPG`, `JPEG 2000`, `PNG`
  - `HDR (30-bit)` — `JPEG 2000` (10-bit-per-channel encoding)
- **Name Suffix:** Customize the output filename suffix (defaults to `-compressed`; leave blank for safe atomic in-place overwrite).
- **Quality Factor (`1–100`) & Resolution Scale (`0.10–1.00`):** Fine-grained `0.01` resolution scaling with instant O(1) empirical size estimation.
- **3-State Target Size Solver:** Cycles between `Target Size` (`Off`, showing inline `Estimated Size`), `Target Size (MB):`, and `Target Size (%):`, automatically solving `Resolution Scale` first (`1.00` down to `0.10`) and then `Quality Factor` (`100` down to `1`) on lossy formats.

### Interactive Full-Page Visual Diff Preview
- **Real-Time Full-Page Rasterization:** Powered by `hayro`, displaying the full composed page (vector text, graphics, and re-encoded images) rather than isolated image streams.
- **Three-Zone Header Toolbar:**
  - **Left:** `Page Preview` title and live rendering spinner.
  - **Center:** Dynamically centered `<  [ X / Y ]  >` page navigation capsule with inline editable page jump.
  - **Right:** Right-aligned `[100%]   [Fit]   [Original]` controls — fixed `58 × 22 px` inline-editable custom zoom pill (`10%–500%`), `Fit` / `Fill` / `Center` framing cycle, and a **press-and-hold `Original`** button that instantaneously flips to the uncompressed original page texture at the exact same zoom level and scroll offset.
- **Instant Navigation Gestures:** Zero-animation, cursor-centered trackpad pinch-to-zoom and `Ctrl + Scrollwheel` zooming (`10%–500%`), plus left-click drag panning (`Grab` / `Grabbing` cursors).

---

## Optimization Pipeline & Low-End Hardware Architecture

### Multi-Codec Input Decoding
Decodes `/DCTDecode`, `/JPXDecode` (`hayro-jpeg2000`), `/JBIG2Decode` (`hayro-jbig2`), `/CCITTFaxDecode` (`hayro-ccitt`), and raw/deflated streams across `/DeviceGray`, `/DeviceRGB`, `/DeviceCMYK`, `/CalGray`, `/CalRGB`, `/ICCBased`, and `/Indexed` color spaces.

### Structural Optimization & Anti-Bloat Pipeline
1. **Reachability Garbage Collection:** Prunes orphaned incremental-save objects unreachable from `/Root` and `/Info`.
2. **Iterative Object Deduplication:** Uses fast 128-bit structural hashing (`FastHasher128`) to merge duplicate streams and dictionaries before image processing so repeated images are encoded only once.
3. **MRC & 1-Bit Preservation:** Automatically passes through 1-bit bi-level CCITT/JBIG2 streams and tiny icons (`< 50×50` px) untouched; preserves Mixed Raster Content (MRC) `/SMask` streams by scaling 8-bit soft masks losslessly alongside their parent image while keeping 1-bit stencil `/Mask` streams intact.
4. **Per-Stream Size-Bloat Guard:** Retains original image streams whenever re-encoding at `True Color (24-bit)` does not reduce byte size, and passes single-filter `/DCTDecode` streams straight through at `JPG Q100` / `1.00` scale.
5. **Auxiliary Flate Recompression & Modern Container Packing:** Recompresses non-image `/FlateDecode` and raw streams at zlib level 9 (`zlib-rs`) with reusable scratch buffers and writes compact PDF 1.5+ `/ObjStm` Object Streams and binary `/XRef` Cross-Reference Streams (`save_modern`).

### Low-RAM & Low-Level Systems Optimizations
- **Bounded Memory Footprint (`< 150 MB` Peak RAM):** Processes images in bounded chunks sized to worker concurrency, immediately applying encoded streams in place and dropping decoded pixel buffers between chunks; caches only the single active previewed page.
- **Non-Intrusive Background Threading:** Dedicated `rayon` worker pool capped to `50%` of logical cores (`1..=4` threads) running at **Below Normal** OS thread priority so the desktop UI stays responsive on low-core machines.
- **Systems & Compiler Tuning:**
  - `mimalloc` global allocator (`#[global_allocator]`) for fast multi-threaded image allocations and lower heap fragmentation.
  - Boxed large enum/event payloads, SoA numeric batch metrics (`BatchHotMetrics`), `SmallVec` stack-allocated PDF filter/decode chains, `Cow<'a, [u8]>` zero-copy stream borrowing, and non-generic inner delegation to prevent monomorphization bloat.
  - Static chunk slicing (`as_chunks` / `as_chunks_mut`), pre-computed LUTs, and native SIMD code generation (`-C target-cpu=native` in `.cargo/config.toml`).

---

## System Requirements

- **Rust:** Stable Rust toolchain ([rustup.rs](https://rustup.rs/)) with Cargo (for building from source).
- **No External Runtime Dependencies:** Compiles to a standalone native executable.

---

## Build & Run Instructions

1. **Build the optimized release executable:**
   ```powershell
   cargo build --release
   ```
   The `[profile.release]` configuration enables `opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `strip = true`, and the `mimalloc` global allocator. `.cargo/config.toml` also enables `-C target-cpu=native` for local AVX2/SSE4.2 SIMD acceleration (remove that flag if distributing the compiled binary to older CPUs).

2. **Run directly with Cargo:**
   ```powershell
   cargo run --release
   ```

3. **Or launch the compiled binary directly:**
   - **Windows:** `target\release\pdf-compressor.exe`
   - **macOS / Linux:** `target/release/pdf-compressor`

4. **Run the verification suite:**
   ```powershell
   cargo check --all-targets
   cargo clippy --all-targets -- -W clippy::all -D warnings
   cargo test
   ```

> **Note:** Password-protected / encrypted PDFs cannot be unlocked automatically and will report an error upon loading or compression.

---

## File Structure

```text
PDF-Image-Compressor-main/
├── .cargo/
│   └── config.toml         # Native CPU SIMD rustflags (-C target-cpu=native)
├── src/
│   ├── main.rs             # Entry point, mimalloc global allocator & eframe viewport setup
│   ├── app.rs              # egui/eframe UI, unified file table, preview toolbar/zoom/pan & workers
│   ├── pdf_processor.rs    # PDF inspection, GC/dedup, codec decoding, estimator, preview & compression
│   ├── jpeg_enc.rs         # MozJPEG encoder/decoder (trellis, progressive, CMYK->RGB)
│   └── jpx_enc.rs          # Pure-Rust ITU-T T.800 JPEG 2000 Part 1 encoder (/JPXDecode, 1/8/10-bit)
├── Cargo.toml              # Crate dependencies & fat-LTO release profile
├── Cargo.lock              # Locked dependency graph
└── README.md               # Project documentation
```

---

## Future Improvements

1. **Optional Encryption & Metadata Controls:** Add user-password unlocking prompts and optional stripping/editing of document metadata (`/Info` and XMP).
2. **Standalone Image & Document Support:** Extend beyond PDFs to support compressing standalone image files and additional document formats.
