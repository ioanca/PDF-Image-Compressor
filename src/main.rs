#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod app;
mod pdf_processor;

use app::PdfCompressorApp;
use eframe::egui;

fn main() -> eframe::Result {
    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([640.0, 680.0])
            .with_min_inner_size([520.0, 560.0])
            .with_drag_and_drop(true),
        ..Default::default()
    };

    eframe::run_native(
        "PDF Image Compressor",
        native_options,
        Box::new(|cc| Ok(Box::new(PdfCompressorApp::new(cc)))),
    )
}
