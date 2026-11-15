# PDF Image Compressor

A fast, lightweight desktop GUI application written in Rust for compressing PDF files containing scanned pages, documents, or photos while preserving physical page dimensions (`/MediaBox`, `/CropBox`) and visual layout (`cm` transformation matrices).

It features a native single-executable interface with single and batch PDF processing, live O(1) output size estimation, automatic target-size solving, multi-core image re-encoding (`JPG`, `JPEG 2000`, `PNG`), and lossless PDF structural optimization.

## Project Overview

The PDF Image Compressor is a desktop tool built to shrink bloated PDFs without distorting page geometry or breaking reader compatibility. The application parses unencrypted PDF documents in memory using `lopdf` and automatically:

- Inspects pages, embedded Image XObjects, and structural overhead upon file load
- Estimates compressed file sizes in real time as format, quality, scale, or filter settings change
- Solves optimal scale and quality settings to hit user-defined MB or percentage targets
- Re-encodes embedded images across all CPU cores while leaving small icons, 1-bit scans, and bloated outputs untouched
- Prunes unreachable objects, deduplicates identical streams, re-compresses `/FlateDecode` streams, and writes compact PDF 1.5+ object/XRef streams

### Key Features

- **Native GUI & Drag-and-Drop**: Built with `eframe`/`egui` and `rfd` (`Select PDF` supports single or multi-file selection, alongside drag-and-drop for `.pdf` files)
- **Three PDF-Native Output Codecs**:
  - **JPG (`/DCTDecode`)**: Powered by MozJPEG (`mozjpeg`) with trellis quantization, progressive scan optimization, overshoot deringing, and adaptive `4:2:0` (`quality <= 85`) / `4:4:4` (`quality > 85`) chroma subsampling
  - **JPEG 2000 (`/JPXDecode`)**: Built-in ITU-T T.800 Part 1 encoder (`jpx_enc`) using Irreversible Color Transform (ICT), CDF 9/7 DWT, and EBCOT arithmetic coding
  - **PNG (`/FlateDecode`)**: Lossless encoding with adaptive PNG row filtering (PDF `/Predictor 15`) and zlib fallback
- **Batch Processing & Multi-Selection**: Load multiple PDFs into an interactive table (`Click`, `Ctrl/Cmd+Click`, `Shift+Click`, `Ctrl/Cmd+A`), toggle `Est. (MB)` / `Est. (%)` previews, configure files individually or via `Apply for ALL`, and export to a chosen folder
- **3-State Target Size Solver**: Cycles between `Target Size` (off, showing inline `Estimated Size`), `Target Size (MB):`, and `Target Size (%):`, automatically adjusting resolution scale and quality factor
- **Multi-Codec Input Decoding**: Decodes `/DCTDecode`, `/JPXDecode` (`hayro-jpeg2000`), `/JBIG2Decode` (`hayro-jbig2`), `/CCITTFaxDecode` (`hayro-ccitt`), and raw/deflated streams across `/DeviceGray`, `/DeviceRGB`, `/DeviceCMYK`, `/CalGray`, `/CalRGB`, `/ICCBased`, and `/Indexed` color spaces
- **Structural Optimization & Anti-Bloat Guards**: Performs unreachable-object garbage collection, iterative stream/object deduplication, `/FlateDecode` level-9 re-compression of non-image streams, and PDF 1.5+ `/ObjStm` + binary `/XRef` packing

## System Requirements

- **Rust**: Stable Rust toolchain ([rustup.rs](https://rustup.rs/)) with Cargo (for building from source)
- **Windows GNU Toolchain (optional)**: MinGW-w64 (`gcc`, `ar`, `dlltool`) if targeting `x86_64-pc-windows-gnu` as configured in `.cargo/config.toml`
- **Core Rust Crates** (managed via `Cargo.lock`): `eframe`/`egui`, `rfd`, `lopdf`, `image`, `mozjpeg`, `flate2`, `rayon`, `sha2`, `thiserror`, `hayro-ccitt`, `hayro-jbig2`, and `hayro-jpeg2000`
- **No External Runtime Dependencies**: Compiles to a standalone native executable

## Installation & Setup

1. **Navigate to the project directory:**

2. **Build the standalone release executable:**
   ```powershell
   cargo build --release
   ```

3. **Run the application:**
   ```powershell
   cargo run --release
   ```
   Or launch the compiled binary directly:
   - **Windows:** `target\release\pdf-compressor.exe`
   - **macOS / Linux:** `target/release/pdf-compressor`

This will open the **PDF Image Compressor** desktop window.

## Usage

1. Launch the application (`cargo run --release` or `pdf-compressor.exe`).
2. Load one or more unencrypted `.pdf` files using **Select PDF** or by dragging and dropping files onto the window.
3. Configure **Compression & Resize Settings** (in multi-file mode, select specific rows or enable **Apply for ALL**):
   - **Image Format**: Choose `JPG`, `JPEG 2000`, or `PNG` (lossless; disables **Quality Factor**)
   - **Quality Factor**: Adjust `1..=100` for `JPG` and `JPEG 2000`
   - **Resolution Scale**: Adjust `0.10x..=1.00x` to downscale image pixel dimensions
   - **Resize Filter**: Choose `Fancy` (`Lanczos3`) or `Fast` (`CatmullRom`)
   - **Target Size**: Click the checkbox to cycle `Off` → `Target Size (MB):` → `Target Size (%):`
4. Click **Compress & Save**:
   - **Single-file mode**: Choose an output file via **Save As** (defaults to `<name>_compressed.pdf`). The **Compression Summary** card displays **Original Size**, **New Size**, **Space Saved**, and a **COMPRESSED**, **UNDER TARGET**, or **OVER TARGET** badge.
   - **Batch mode**: Choose an output folder to save `<name>_compressed.pdf` (automatically appending `_2`, `_3`, etc. on name collisions). The summary table highlights each row (`Failed`, over target, or under target) and displays the output folder path.

> **Note:** Password-protected / encrypted PDFs cannot be unlocked automatically and will report an error upon loading or compression.

## Base Functionality

### Overview
When PDFs are loaded, `inspect_pdf` parses each document in a background worker thread, runs structural garbage collection and deduplication in memory, and builds a `PdfSummary` without modifying the source files on disk.

The UI adapts dynamically between single-file and multi-file layouts, auto-fitting window height while keeping controls responsive during background inspection and compression via channel-based worker events (`WorkerEvent`).

## Compression & Target-Size Engine

### Overview
The compression pipeline combines fast analytical size estimation with parallel image re-encoding and PDF container compaction.

### How It Works

#### 1. **Bitrate Calibration & O(1) Size Estimation**
- During `inspect_pdf`, the largest compressible image (downscaled to at most ~1 MP if needed) is sampled to calibrate actual `JPG` (at Q50) and `FlateDecode` bytes-per-pixel rates
- Combines calibrated pixel bitrates with structural baselines (`ObjStm`/`XRef` compression factors and `/FlateDecode` stream estimates) so dragging any slider updates **Estimated Size** instantaneously

#### 2. **Target-Size Solver (`scale` First, Then `quality`)**
- When **Target Size (MB)** or **Target Size (%)** is active, `apply_target_scale_first` searches `scale` from `1.00x` down to `0.10x` to find the highest resolution that fits the byte budget
- If `scale` reaches `0.10x` while still over budget—or stays at `1.00x` with spare budget—on a lossy format (`JPG` or `JPEG 2000`), `solve_quality_for_target` searches `quality` (`100` down to `1`) to meet the target
- Adjusting **Quality Factor** manually re-solves **Resolution Scale**, and adjusting **Resolution Scale** manually re-solves **Quality Factor**

#### 3. **Parallel Image Processing & PDF Serialization**
- **Candidate Filtering**: Skips 1-bit bi-level images, images smaller than `50x50` px, and explicit chroma-key `/Mask` arrays; passes through single-filter `/DCTDecode` streams unchanged when `JPG` is set to `Q100` at `1.00x`
- **Mask Handling**: Scales 8-bit `/SMask` alpha streams losslessly alongside their parent image when `scale < 1.0`, while keeping 1-bit `/SMask` and stencil `/Mask` streams intact
- **Anti-Bloat & Container Packing**: Decodes, resizes, and re-encodes images in parallel with `rayon`, retains original image streams whenever re-encoding is not smaller, re-compresses auxiliary `/FlateDecode` streams at level 9, runs post-compression GC, and writes PDF 1.5+ object and binary `/XRef` streams

### File Structure

```text
Popa/
├── .cargo/
│   └── config.toml         # Windows GNU MinGW linker & mozjpeg-sys CFLAGS config
├── src/
│   ├── main.rs             # Application entry point & eframe viewport setup
│   ├── app.rs              # egui/eframe UI, batch selection, target-size controls & workers
│   ├── pdf_processor.rs    # PDF inspection, GC/dedup, codec decoding, estimator & compression
│   ├── jpeg_enc.rs         # MozJPEG encoder/decoder (trellis, progressive, CMYK->RGB)
│   └── jpx_enc.rs          # Pure-Rust ITU-T T.800 JPEG 2000 Part 1 encoder (/JPXDecode)
├── Cargo.lock              # Locked dependency graph for pdf-compressor v0.1.0
└── README.md               # Project documentation
```

### Future Improvements

1. **Optional Encryption & Metadata Controls**: Add user-password unlocking prompts and optional stripping/editing of document metadata (`/Info` and XMP).
2. **Per-Page Preview & Visual Diff**: Render side-by-side page thumbnails comparing original vs. compressed image quality before saving.
3. **CLI & Watch-Folder Mode**: Expose a headless command-line mode for automated batch compression workflows without launching the GUI.
