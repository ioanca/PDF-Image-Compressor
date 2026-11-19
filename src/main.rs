#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

mod app;
mod jpeg_enc;
mod jpx_enc;
mod pdf_processor;

use app::{
    PdfCompressorApp, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, MIN_WINDOW_HEIGHT,
    MIN_WINDOW_WIDTH,
};
use eframe::egui;

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT])
            .with_min_inner_size([MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "PDF Image Compressor",
        native_options,
        Box::new(|cc| Ok(Box::new(PdfCompressorApp::new(cc)))),
    )
}
