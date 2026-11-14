# PDF Image Compressor

A fast, lightweight, single-executable desktop GUI application in Rust for shrinking PDF files containing scanned pages or homework photos while keeping the original physical page dimensions (`/MediaBox`, `/CropBox`) and visual layout (`cm` transformation matrices) completely intact.

## Features

- **Native GUI & File Dialogs:** Built with `eframe`/`egui` and `rfd` (plus drag-and-drop `.pdf` support).
- **One-Click Pipeline:** Select a PDF, pick your target format/quality/scale, and click **Compress & Save As...** (defaults to `<original_name>_compressed.pdf`).
- **Toggleable Target Size & Live Estimated Size:** Displays an instant `Estimated Size` readout while unchecked, or checks **Target Size (MB)** to automatically couple Quality and Resolution Scale inversely toward your exact MB budget, with **COMPRESSED** / **UNDER TARGET** / **OVER TARGET** summary feedback.
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
