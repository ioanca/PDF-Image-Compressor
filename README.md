# PDF Image Compressor

A fast, lightweight, single-executable desktop GUI application written in Rust for compressing PDF files containing scanned pages, documents, or photos while preserving exact physical page dimensions (`/MediaBox`, `/CropBox`) and visual layout (`cm` transformation matrices).

Built with `eframe`/`egui` (`glow` OpenGL backend), `lopdf`, `hayro`, `mozjpeg`, `flate2` (`zlib-rs`), `rayon`, `smallvec`, and `mimalloc`, it combines a unified single- and multi-file explorer interface, real-time empirical size estimation, a 3-state target-size solver, an interactive full-page visual diff preview, and a low-RAM, crash-hardened PDF optimization pipeline.

---

## Key Features

### Unified Single-File & Batch Interface
- **Explorer-Style File Table:** Load one or more unencrypted `.pdf` files via **Select** (`rfd`) or drag-and-drop. Single-file and multi-file sessions share the exact same unified layout with multi-row selection (`Click`, `Ctrl/Cmd+Click`, `Shift+Click`, `Ctrl/Cmd+A`), **Apply for ALL** (shown when `> 1` file is loaded while reserving its row height so **Show Preview** never shifts vertically), **Show Preview**, and **Compress**.
- **Clickable Estimation Toggle:** Click the `Est. (MB)` / `Est. (%)` table header to switch between estimated output size in megabytes and percentage of original size.
- **No Outer Scrollbar & Display-Aware Table Scaling:**
  - Vertical scrolling is confined strictly to the inner tables — there is no outer main-panel scrollbar.
  - The top file list caps at **10 visible rows**.
  - **Compression Summary** stays hidden when 0 files are loaded, fits exact `1–4` rows cleanly when `< 5` files are loaded, and scales dynamically by integer rows between **5 and 9 visible rows** on `≤ 1080p` displays (up to **10 rows** on `1440p`/`4K` displays) so rows never clip mid-line.
  - A fixed `22 px` **`"Saved to: ..."`** footer band (with path truncation) is always reserved at the bottom of **Compression Summary** so completing compression never shifts card height or overflows the window.
- **DPI-Aware Viewport & Resize Locking:**
  - The main control panel locks to `27.5%` of logical monitor width (vertical-only window resizing when preview is closed).
  - Opening **Show Preview** expands the window by `42.5%` (`70%` total width) and unlocks horizontal resizing for the preview pane only.
  - Dynamic `MinInnerSize` adapts automatically to 0 files, 1–4 files, and 5+ files so the bottom window border sits cleanly underneath the cards.

### Compression Settings
All controls live in a fixed-column grid (`158 px` label column, aligned slider tracks, `64 px` numeric boxes) so toggling options never shifts widgets:

1. **Resize Filter:** `Fancy` (`Lanczos3`) for sharper downsampling or `Fast` (`CatmullRom`) for quicker resizing.
2. **Image Format (PDF-Native Output Codecs):**
   - **JPG (`/DCTDecode`):** MozJPEG (`mozjpeg`) with trellis quantization, progressive scan optimization, overshoot deringing, adaptive `4:2:0` (`quality ≤ 85`) / `4:4:4` (`quality > 85`) chroma subsampling for RGB, and a hardened fast-safe path for `Q ≥ 98` (`Q = 100`) and narrow strips (`< 8 px`, routed to lossless `/FlateDecode` in PDF streams).
   - **JPEG 2000 (`/JPXDecode`):** Pure-Rust ITU-T T.800 Part 1 encoder (`jpx_enc`) with Irreversible Color Transform (ICT), CDF 9/7 DWT, and EBCOT arithmetic coding.
   - **PNG (`/FlateDecode`):** Lossless encoding using adaptive PNG row filtering (PDF `/Predictor 15`), raw zlib fallback, and 1-bit bi-level packing.
3. **Format-Adaptive Color Depth:** Automatically filters to depths supported by the selected format and falls back cleanly on codec switches:
   - `Black/White (1-bit)` — `JPEG 2000`, `PNG`
   - `Grayscale (8-bit)` — `JPG`, `JPEG 2000`, `PNG`
   - `High Color (16-bit)` — `JPEG 2000`, `PNG` (RGB565 quantization)
   - `True Color (24-bit)` — `JPG`, `JPEG 2000`, `PNG`
   - `HDR (30-bit)` — `JPEG 2000` (10-bit-per-channel encoding)
4. **Name Suffix:** Custom output filename suffix (defaults to `-compressed`; leave blank to safely overwrite the original filename in-place). Automatically strips Windows-forbidden characters (`\ / : * ? " < > |`, control chars) and trailing spaces/dots.
5. **Quality Factor (`1–100`) & Resolution Scale (`0.10–1.00`):** Fine-grained `0.01` resolution scaling with instant O(1) empirical size estimation.
6. **3-State Target Size Solver:** Cycles `Target Size` (`Off`, showing inline `Estimated Size`) → `Target Size (MB):` → `Target Size (%):`, solving `Resolution Scale` first (`1.00` down to `0.10`) and then `Quality Factor` (`100` down to `1`) on lossy formats.

### Interactive Full-Page Visual Diff Preview
- **Full-Page Rasterization (`hayro`):** Renders the complete composed page (vector text, graphics, and re-encoded images) rather than isolated image streams, showing the compressed result by default.
- **Fixed-Height Three-Zone Header Bar (`24 px`):**
  - **Left:** `Page Preview` title and live rendering spinner.
  - **Center:** Dynamically centered `<  [ X / Y ]  >` page navigation capsule with inline editable page jump (clamped cleanly on narrow widths).
  - **Right:** Right-anchored `[100%]   [Fit]   [Original]` toolbar:
    - **`[100%]` Zoom Pill:** Custom-painted fixed `58 × 22 px` hitbox with a clipped inline frameless editor (`10%–500%`) so clicking to edit never shifts neighboring buttons or the preview top margin.
    - **`[Fit]` Framing Cycle:** Cycles `Fit` → `Fill` → `Center`.
    - **`[Original]` Hold-to-Compare:** Press and hold to instantaneously swap to the uncompressed original page texture at the exact same zoom level and scroll offset; release to return to the compressed preview.
- **Zero-Animation Gestures & Panning:** Instant cursor-centered trackpad pinch-to-zoom and `Ctrl + Scrollwheel` zooming (`10%–500%`), plus left-click drag panning (`Grab` / `Grabbing` cursors).
- **Single-Page Texture Cache:** Retains textures only for the active previewed page and evicts them immediately when switching pages/files or unchecking **Show Preview**.

---

## Usage

1. Launch the app (`cargo run --release` or `pdf-compressor.exe`).
2. Click **Select** or drag-and-drop one or more unencrypted `.pdf` files onto the window.
3. Adjust settings in the **Compression** card (select individual rows or enable **Apply for ALL** when multiple files are loaded).
4. Optionally check **Show Preview** to inspect pages, zoom/pan, and press-and-hold **Original** to compare before/after quality.
5. Click **Compress** and choose an output folder:
   - Each queued file's path and settings (including **Name Suffix**) are snapshotted at click time, and concurrent clicks are ignored while compressing.
   - Output is always written via `BufWriter` to a unique sibling temporary file (`.tmp_pdfcomp_<pid>_<index>.pdf`), flushed, synced, and atomically moved into place. If a target PDF is open in another viewer (e.g., Acrobat, Edge, Chrome), the app reports `"File is open in another program"` cleanly without corrupting the file.
   - **Compression Summary** updates row-by-row with **Original Size**, **New Size**, **Space Saved**, color-coded target status, and the final **`Saved to: ...`** path.

> **Note:** Password-protected / encrypted PDFs cannot be unlocked automatically and will report an error upon loading or compression.

---

## Optimization Pipeline & Systems Architecture

### Multi-Codec Input Decoding
Decodes `/DCTDecode`, `/JPXDecode` (`hayro-jpeg2000`), `/JBIG2Decode` (`hayro-jbig2`), `/CCITTFaxDecode` (`hayro-ccitt`), and raw/deflated streams across `/DeviceGray`, `/DeviceRGB`, `/DeviceCMYK`, `/CalGray`, `/CalRGB`, `/ICCBased`, and `/Indexed` color spaces.

### Structural Optimization & Anti-Bloat Pipeline
1. **Reachability Garbage Collection:** Prunes orphaned incremental-save objects unreachable from `/Root` and `/Info`.
2. **Iterative Object Deduplication:** Merges duplicate streams and dictionaries via fast 128-bit structural hashing (`FastHasher128`) before image processing so repeated images are encoded once.
3. **MRC & 1-Bit Preservation:** Leaves 1-bit bi-level CCITT/JBIG2 streams and tiny icons (`< 50×50` px) untouched; scales 8-bit Mixed Raster Content (MRC) `/SMask` streams losslessly alongside their parent image while keeping 1-bit stencil `/Mask` streams intact.
4. **Per-Stream Anti-Bloat Guard:** Retains original image streams whenever `True Color (24-bit)` re-encoding is not smaller, and passes single-filter `/DCTDecode` streams straight through at `JPG Q100` / `1.00` scale.
5. **Auxiliary Flate Recompression & Modern Packing:** Recompresses non-image `/FlateDecode` and raw streams at zlib level 9 (`zlib-rs`) using reusable scratch buffers and writes compact PDF 1.5+ `/ObjStm` Object Streams and binary `/XRef` Cross-Reference Streams (`save_modern`).

### Resource Management & Low-Level Rust Optimizations
- **Bounded CPU & RAM Footprint (`< 150 MB` Peak RAM):**
  - Single global `OnceLock` `rayon::ThreadPool` capped to `50%` of logical cores (`1..=3` threads), `8 MiB` worker stacks, and **Below Normal** OS thread priority (`THREAD_PRIORITY_BELOW_NORMAL` on Windows, `nice(10)` on Unix).
  - Processes PDFs strictly one at a time sequentially, chunks image and Flate work to `pool.current_num_threads()`, drops decoded pixel buffers immediately after each chunk, and runs cooperative inter-chunk pacing (`yield_now()` + `2 ms` sleep) to prevent Task Manager CPU spikes.
  - Zero `.unwrap()` / `.expect()` calls across all non-test production code (including graceful sequential fallback if thread-pool creation fails).
- **Compiler & Memory-Layout Tuning:**
  - `mimalloc` global allocator (`#[global_allocator]`) for fast multi-threaded allocations and low heap fragmentation.
  - Boxed large enum/event payloads (`PreviewCommand`, `WorkerEvent`, `ImageCompressionOutcome`, `PreviewDisplay`), SoA numeric batch metrics (`BatchHotMetrics`), `SmallVec` stack-allocated PDF filter/decode chains, `Cow<'a, [u8]>` zero-copy stream borrowing, reusable `FlateScratch`/`ImageEncodeScratch` buffers, static `as_chunks`/`as_chunks_mut` SIMD-friendly pixel loops, compile-time LUTs, `#[cold]` error helpers, and non-generic inner delegation to prevent monomorphization bloat.
  - Release profile (`opt-level = 3`, `lto = "fat"`, `codegen-units = 1`, `strip = true`) paired with `.cargo/config.toml` (`-C target-cpu=native`).

---

## System Requirements

- **Rust:** Stable Rust toolchain ([rustup.rs](https://rustup.rs/)) with Cargo (for building from source).
- **No External Runtime Dependencies:** Compiles to a standalone native executable.

---

## Build, Run & Test

1. **Build the optimized release executable:**
   ```powershell
   cargo build --release
   ```
   > `.cargo/config.toml` enables `-C target-cpu=native` for local AVX2/SSE4.2 SIMD acceleration. Remove that flag if distributing the compiled binary to older CPUs.

2. **Run directly with Cargo:**
   ```powershell
   cargo run --release
   ```

3. **Or launch the compiled binary directly:**
   - **Windows:** `target\release\pdf-compressor.exe`
   - **macOS / Linux:** `target/release/pdf-compressor`

4. **Run the verification suite (check, Clippy, and all 45 unit/integration tests):**
   ```powershell
   cargo check --all-targets
   cargo clippy --all-targets -- -W clippy::all -D warnings
   cargo test
   ```

---

## File Structure

```text
PDF-Image-Compressor-main/
├── .cargo/
│   └── config.toml         # Native CPU SIMD rustflags (-C target-cpu=native)
├── src/
│   ├── main.rs             # Entry point, mimalloc global allocator & eframe viewport setup
│   ├── app.rs              # egui/eframe UI, unified tables, zoom/pan preview, safe suffix/save & workers
│   ├── pdf_processor.rs    # PDF inspection, GC/dedup, codec decoding, estimator, preview & compression
│   ├── jpeg_enc.rs         # Hardened MozJPEG encoder/decoder (trellis, progressive, Q=100/strip safety, CMYK->RGB)
│   └── jpx_enc.rs          # Pure-Rust ITU-T T.800 JPEG 2000 Part 1 encoder (/JPXDecode, 1/8/10-bit)
├── Cargo.toml              # Crate dependencies & fat-LTO release profile
├── Cargo.lock              # Locked dependency graph
└── README.md               # Project documentation
```

---

## Future Improvements

1. **Optional Encryption & Metadata Controls:** Add user-password unlocking prompts and optional stripping/editing of document metadata (`/Info` and XMP).
2. **Standalone Image & Document Support:** Extend beyond PDFs to support compressing standalone image files and additional document formats.
