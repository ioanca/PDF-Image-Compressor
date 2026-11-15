# PDF Image Compressor

A fast, lightweight, single-executable desktop GUI application in Rust for shrinking PDF files containing scanned pages or homework photos while keeping the original physical page dimensions (`/MediaBox`, `/CropBox`) and visual layout (`cm` transformation matrices) completely intact.

## Features

- **Native GUI & Batch File Support:** Built with `eframe`/`egui` and `rfd` (`Select PDF` supports single or multi-file selection, plus drag-and-drop for one or multiple `.pdf` files).
- **Single & Batch Export:** Select a single PDF to pick an output file via **Save As** (defaulting to `<original_name>_compressed.pdf`), or load multiple PDFs to export all of them into a chosen destination folder (`<original_stem>_compressed.pdf`).
- **Per-File & Shared Settings (`Apply for ALL`):** Configure format (`JPEG`, `PNG`, `WebP`, `AVIF`), quality, resolution scale, resize filter (`Fancy` / `Fast`), and a 3-state **Target Size** cycle (`Off` → `Target Size (MB)` → `Target Size (%)` with inverse Quality/Scale coupling) per file, across multi-selected rows (`Click`, `Ctrl/Cmd+Click`, `Shift+Click`, `Ctrl/Cmd+A`), or across all loaded files via **Apply for ALL**.
- **Always-Visible Compression Summary:** In single-file mode, displays **Original Size**, **New Size**, and **Space Saved** cleanly (showing estimated percentage saved prior to export and `-` before load), with **COMPRESSED** / **UNDER TARGET** / **OVER TARGET** badge states. In multi-file mode, shows an Explorer-aligned 4-column per-row summary (**No.**, **Original Size**, **New Size**, **Space Saved**) with green/red row highlights and the saved destination folder path.
- **PDF Format & Compatibility:**
  - **JPEG (`/DCTDecode`)**: Recommended for scanned homework and photos.
  - **PNG (`/FlateDecode` with PNG Predictor 15)**: Lossless PDF-native stream compression.
  - **WebP / AVIF**: Pure-Rust `image-webp` / `ravif` support (no NASM or C libraries needed) with automatic transcoding into standard PDF-compatible `/FlateDecode` or `/DCTDecode` streams so Chrome, Adobe Acrobat, and Apple Preview open the output without errors.
- **Layout Preservation:** Updates only Image XObject pixel dimensions (`/Width`, `/Height`), color space metadata (`/ColorSpace`, `/BitsPerComponent`), and encoded stream data while leaving `/MediaBox`, `/CropBox`, and content stream `cm` matrices untouched.
- **Edge-Case Handling:** Skips icons under `50x50` pixels and `/ImageMask true` objects, handles `/DeviceRGB`, `/DeviceGray`, `/DeviceCMYK`, `/ICCBased`, and `/Indexed` color spaces, composites `/SMask` alpha channels over white, and removes conflicting `/DecodeParms` / `/Decode` / `/SMask` entries.

## Build Instructions

Requires the latest stable Rust toolchain ([rustup.rs](https://rustup.rs/)). No external C libraries or `nasm` assembler are required.

```powershell
# Build the standalone release executable
cargo build --release

# Optionally run the unit tests
cargo test
```

After `cargo build --release` finishes, the standalone executable is located at:

- **Windows:** `target\release\pdf-compressor.exe`
- **macOS / Linux:** `target/release/pdf-compressor`
