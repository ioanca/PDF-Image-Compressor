use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

use eframe::egui::{self, Color32, CornerRadius, Frame, Margin, RichText, Stroke};
use rfd::FileDialog;

use crate::pdf_processor::{
    compress_pdf, inspect_pdf, CompressionOptions, CompressionStats, ImageFormatChoice,
    PdfSummary, ProgressUpdate, ResizeFilterChoice,
};

struct LoadedPdf {
    path: Option<PathBuf>,
    file_name: String,
    bytes: Arc<Vec<u8>>,
    summary: PdfSummary,
}

enum WorkerEvent {
    Progress(ProgressUpdate),
    Finished {
        format_name: &'static str,
        result: Result<(Vec<u8>, CompressionStats), String>,
    },
}

struct PendingSave {
    bytes: Vec<u8>,
    stats: CompressionStats,
    format_name: &'static str,
    default_file_name: String,
    initial_dir: Option<PathBuf>,
}

struct CompletedResult {
    saved_path: PathBuf,
    stats: CompressionStats,
    format_name: &'static str,
}

pub struct PdfCompressorApp {
    loaded: Option<LoadedPdf>,
    options: CompressionOptions,
    worker_rx: Option<Receiver<WorkerEvent>>,
    progress: Option<ProgressUpdate>,
    last_result: Option<CompletedResult>,
    status_message: Option<String>,
    error_message: Option<String>,
}

impl Default for PdfCompressorApp {
    fn default() -> Self {
        Self {
            loaded: None,
            options: CompressionOptions::default(),
            worker_rx: None,
            progress: None,
            last_result: None,
            status_message: None,
            error_message: None,
        }
    }
}

impl PdfCompressorApp {
    pub fn new(_cc: &eframe::CreationContext<'_>) -> Self {
        Self::default()
    }

    fn is_busy(&self) -> bool {
        self.worker_rx.is_some()
    }

    fn open_file_dialog(&mut self) {
        let mut dialog = FileDialog::new().add_filter("PDF Document", &["pdf"]);
        if let Some(dir) = self
            .loaded
            .as_ref()
            .and_then(|l| l.path.as_ref())
            .and_then(|p| p.parent())
        {
            dialog = dialog.set_directory(dir);
        }

        if let Some(path) = dialog.pick_file() {
            self.load_pdf_from_path(path);
        }
    }

    fn load_pdf_from_path(&mut self, path: PathBuf) {
        self.error_message = None;
        self.status_message = None;
        self.last_result = None;

        match std::fs::read(&path) {
            Ok(bytes) => {
                let file_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "document.pdf".to_string());
                self.load_pdf_bytes(Some(path), file_name, bytes);
            }
            Err(err) => {
                self.error_message = Some(format!("Could not read file: {err}"));
            }
        }
    }

    fn load_pdf_bytes(&mut self, path: Option<PathBuf>, file_name: String, bytes: Vec<u8>) {
        match inspect_pdf(&bytes) {
            Ok(summary) => {
                self.loaded = Some(LoadedPdf {
                    path,
                    file_name,
                    bytes: Arc::new(bytes),
                    summary,
                });
                self.error_message = None;
                self.last_result = None;
                self.status_message = None;
            }
            Err(err) => {
                self.error_message = Some(err.to_string());
            }
        }
    }

    fn start_compression(&mut self, ctx: &egui::Context) {
        let Some(loaded) = &self.loaded else {
            return;
        };

        self.error_message = None;
        self.status_message = None;
        self.last_result = None;

        let bytes = Arc::clone(&loaded.bytes);
        let options = self.options.clone();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress = Some(ProgressUpdate {
            current: 0,
            total: loaded.summary.image_count.max(1),
            message: "Starting PDF optimization...".to_string(),
        });

        let repaint_ctx = ctx.clone();
        let format_name = options.format.short_name();
        thread::spawn(move || {
            let result = compress_pdf(&bytes, &options, |update| {
                let _ = tx.send(WorkerEvent::Progress(update));
                repaint_ctx.request_repaint();
            })
            .map_err(|e| e.to_string());

            let _ = tx.send(WorkerEvent::Finished {
                format_name,
                result,
            });
            repaint_ctx.request_repaint();
        });
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker_rx else {
            return;
        };

        let mut finished_payload: Option<(&'static str, Result<(Vec<u8>, CompressionStats), String>)> =
            None;
        while let Ok(event) = rx.try_recv() {
            match event {
                WorkerEvent::Progress(p) => {
                    self.progress = Some(p);
                }
                WorkerEvent::Finished {
                    format_name,
                    result,
                } => {
                    finished_payload = Some((format_name, result));
                }
            }
        }

        if let Some((format_name, result)) = finished_payload {
            self.worker_rx = None;
            self.progress = None;

            match result {
                Ok((out_bytes, stats)) => {
                    let pending = self.build_pending_save(out_bytes, stats, format_name);
                    self.prompt_save_as(pending);
                }
                Err(err) => {
                    self.error_message = Some(err);
                }
            }
        }
    }

    fn build_pending_save(
        &self,
        bytes: Vec<u8>,
        stats: CompressionStats,
        format_name: &'static str,
    ) -> PendingSave {
        let (default_file_name, initial_dir) = match &self.loaded {
            Some(loaded) => {
                let stem = loaded
                    .path
                    .as_deref()
                    .and_then(Path::file_stem)
                    .or_else(|| Path::new(&loaded.file_name).file_stem())
                    .map(|s| s.to_string_lossy().into_owned())
                    .filter(|s| !s.is_empty())
                    .unwrap_or_else(|| "document".to_string());
                let dir = loaded
                    .path
                    .as_ref()
                    .and_then(|p| p.parent().map(Path::to_path_buf));
                (format!("{stem}_compressed.pdf"), dir)
            }
            None => ("compressed.pdf".to_string(), None),
        };

        PendingSave {
            bytes,
            stats,
            format_name,
            default_file_name,
            initial_dir,
        }
    }

    fn prompt_save_as(&mut self, pending: PendingSave) {
        let mut dialog = FileDialog::new()
            .add_filter("PDF Document", &["pdf"])
            .set_file_name(&pending.default_file_name);

        if let Some(dir) = &pending.initial_dir {
            dialog = dialog.set_directory(dir);
        }

        match dialog.save_file() {
            Some(save_path) => match std::fs::write(&save_path, &pending.bytes) {
                Ok(()) => {
                    self.status_message = Some(format!(
                        "Saved compressed PDF to {}",
                        save_path.display()
                    ));
                    self.last_result = Some(CompletedResult {
                        saved_path: save_path,
                        stats: pending.stats,
                        format_name: pending.format_name,
                    });
                }
                Err(err) => {
                    self.error_message =
                        Some(format!("Compression succeeded, but saving failed: {err}"));
                }
            },
            None => {
                self.status_message = Some(format!(
                    "Save cancelled (compressed output would have been {}).",
                    format_bytes(pending.stats.compressed_bytes)
                ));
            }
        }
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        if self.is_busy() {
            return;
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            let path = file.path().to_path_buf();
            if !path.as_os_str().is_empty() && path.exists() {
                let is_pdf = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
                if is_pdf {
                    self.load_pdf_from_path(path);
                } else {
                    self.error_message = Some(format!(
                        "Unsupported file dropped ({}). Please select a .pdf file.",
                        path.display()
                    ));
                }
                break;
            } else if let Ok(bytes) = file.bytes() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "document.pdf".to_string());
                self.load_pdf_bytes(None, name, bytes);
                break;
            }
        }
    }

    fn render_file_card(&mut self, ui: &mut egui::Ui, busy: bool, is_hovered: bool) {
        let dark = ui.visuals().dark_mode;
        let card_fill = if is_hovered {
            if dark {
                Color32::from_rgb(35, 52, 74)
            } else {
                Color32::from_rgb(232, 242, 255)
            }
        } else if dark {
            Color32::from_rgb(30, 33, 40)
        } else {
            Color32::from_rgb(246, 248, 251)
        };
        let border_color = if is_hovered {
            Color32::from_rgb(66, 133, 244)
        } else if dark {
            Color32::from_rgb(56, 62, 74)
        } else {
            Color32::from_rgb(214, 220, 229)
        };

        Frame::new()
            .fill(card_fill)
            .stroke(Stroke::new(1.0, border_color))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    let select_btn = egui::Button::new(
                        RichText::new("Select PDF...").size(15.0).strong(),
                    )
                    .min_size(egui::vec2(140.0, 36.0));

                    if ui.add_enabled(!busy, select_btn).clicked() {
                        self.open_file_dialog();
                    }

                    ui.add_space(10.0);

                    if let Some(loaded) = &self.loaded {
                        ui.vertical(|ui| {
                            ui.label(RichText::new(&loaded.file_name).strong().size(15.0));
                            ui.label(
                                RichText::new(format!(
                                    "Original size: {}   •   {} page(s)   •   {} image(s) detected",
                                    format_bytes(loaded.summary.file_bytes),
                                    loaded.summary.page_count,
                                    loaded.summary.image_count
                                ))
                                .size(12.5)
                                .color(ui.visuals().weak_text_color()),
                            );
                        });
                    } else {
                        ui.vertical(|ui| {
                            ui.label(RichText::new("No PDF selected").strong().size(14.5));
                            ui.label(
                                RichText::new("Click \"Select PDF...\" or drag & drop a .pdf file here")
                                    .size(12.5)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        });
                    }
                });
            });
    }

    fn render_controls_card(&mut self, ui: &mut egui::Ui, busy: bool) {
        let dark = ui.visuals().dark_mode;
        let card_fill = if dark {
            Color32::from_rgb(30, 33, 40)
        } else {
            Color32::from_rgb(246, 248, 251)
        };
        let border_color = if dark {
            Color32::from_rgb(56, 62, 74)
        } else {
            Color32::from_rgb(214, 220, 229)
        };

        Frame::new()
            .fill(card_fill)
            .stroke(Stroke::new(1.0, border_color))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Compression & Resize Settings").strong().size(15.0));
                ui.add_space(8.0);

                ui.add_enabled_ui(!busy, |ui| {
                    // Format / Algorithm dropdown
                    ui.horizontal(|ui| {
                        ui.label("Image Format:");
                        egui::ComboBox::from_id_salt("image_format_selector")
                            .selected_text(self.options.format.label())
                            .width(360.0)
                            .show_ui(ui, |ui| {
                                for fmt in ImageFormatChoice::ALL {
                                    ui.selectable_value(
                                        &mut self.options.format,
                                        fmt,
                                        fmt.label(),
                                    );
                                }
                            });
                    });

                    if let Some(note) = self.options.format.compatibility_note() {
                        ui.add_space(6.0);
                        let note_fill = if dark {
                            Color32::from_rgb(54, 45, 24)
                        } else {
                            Color32::from_rgb(255, 247, 224)
                        };
                        let note_stroke = Color32::from_rgb(217, 155, 38);
                        Frame::new()
                            .fill(note_fill)
                            .stroke(Stroke::new(1.0, note_stroke))
                            .corner_radius(CornerRadius::same(6))
                            .inner_margin(Margin::same(8))
                            .show(ui, |ui| {
                                ui.set_width(ui.available_width());
                                ui.label(RichText::new(note).size(12.0));
                            });
                    }

                    ui.add_space(10.0);

                    // Quality slider
                    let uses_quality = self.options.format.uses_quality();
                    ui.horizontal(|ui| {
                        ui.label("Quality:");
                        let slider = egui::Slider::new(&mut self.options.quality, 1..=100)
                            .clamping(egui::SliderClamping::Always);
                        ui.add_enabled(uses_quality, slider);
                        if !uses_quality {
                            ui.label(
                                RichText::new("(Lossless mode — ignored)")
                                    .size(12.0)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        }
                    });

                    ui.add_space(8.0);

                    // Resolution Scale slider + quick preset buttons
                    ui.horizontal(|ui| {
                        ui.label("Resolution Scale:");
                        ui.add(
                            egui::Slider::new(&mut self.options.scale, 0.10..=1.00)
                                .step_by(0.05)
                                .fixed_decimals(2)
                                .suffix("x"),
                        );
                    });

                    ui.add_space(4.0);
                    ui.horizontal(|ui| {
                        ui.label(
                            RichText::new("Presets:")
                                .size(12.5)
                                .color(ui.visuals().weak_text_color()),
                        );
                        for (label, value) in [
                            ("0.25x", 0.25_f32),
                            ("0.5x", 0.50_f32),
                            ("0.75x", 0.75_f32),
                            ("1.0x", 1.00_f32),
                        ] {
                            let selected = (self.options.scale - value).abs() < 0.01;
                            if ui.selectable_label(selected, label).clicked() {
                                self.options.scale = value;
                            }
                        }
                    });

                    ui.add_space(8.0);

                    // Resize filter choice
                    ui.horizontal(|ui| {
                        ui.label("Resize Filter:");
                        for filter in ResizeFilterChoice::ALL {
                            let selected = self.options.resize_filter == filter;
                            if ui.selectable_label(selected, filter.label()).clicked() {
                                self.options.resize_filter = filter;
                            }
                        }
                    });
                });

                ui.add_space(8.0);
                ui.label(
                    RichText::new(
                        "Layout preserved: page /MediaBox, /CropBox, and content stream matrices (cm) remain untouched.",
                    )
                    .size(11.5)
                    .color(ui.visuals().weak_text_color()),
                );
            });
    }

    fn render_summary_card(&self, ui: &mut egui::Ui, result: &CompletedResult) {
        let dark = ui.visuals().dark_mode;
        let stats = &result.stats;
        let over_2mb = stats.exceeds_2mb();
        let near_2mb = stats.near_2mb_decimal();

        let (card_fill, border_color) = if over_2mb {
            if dark {
                (Color32::from_rgb(48, 30, 30), Color32::from_rgb(200, 75, 75))
            } else {
                (Color32::from_rgb(255, 243, 243), Color32::from_rgb(217, 83, 79))
            }
        } else if dark {
            (Color32::from_rgb(26, 44, 34), Color32::from_rgb(60, 160, 95))
        } else {
            (Color32::from_rgb(240, 252, 244), Color32::from_rgb(52, 168, 83))
        };

        Frame::new()
            .fill(card_fill)
            .stroke(Stroke::new(1.0, border_color))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Compression Summary").strong().size(15.0));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if over_2mb {
                            let badge_fill = Color32::from_rgb(198, 40, 40);
                            Frame::new()
                                .fill(badge_fill)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("OVER 2.0 MB")
                                            .strong()
                                            .size(12.0)
                                            .color(Color32::WHITE),
                                    );
                                });
                        } else if near_2mb {
                            let badge_fill = Color32::from_rgb(184, 118, 12);
                            Frame::new()
                                .fill(badge_fill)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("NEAR 2.0 MB (>2,000,000 B)")
                                            .strong()
                                            .size(11.5)
                                            .color(Color32::WHITE),
                                    );
                                });
                        } else {
                            let badge_fill = Color32::from_rgb(34, 139, 72);
                            Frame::new()
                                .fill(badge_fill)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new("UNDER 2.0 MB")
                                            .strong()
                                            .size(12.0)
                                            .color(Color32::WHITE),
                                    );
                                });
                        }
                    });
                });

                ui.add_space(8.0);

                egui::Grid::new("summary_metrics_grid")
                    .num_columns(3)
                    .spacing([28.0, 4.0])
                    .show(ui, |ui| {
                        ui.label(
                            RichText::new("Original Size")
                                .size(12.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                        ui.label(
                            RichText::new("New Size")
                                .size(12.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                        ui.label(
                            RichText::new("Space Saved")
                                .size(12.0)
                                .color(ui.visuals().weak_text_color()),
                        );
                        ui.end_row();

                        ui.label(
                            RichText::new(format_bytes(stats.original_bytes))
                                .strong()
                                .size(16.0),
                        );
                        ui.label(
                            RichText::new(format_bytes(stats.compressed_bytes))
                                .strong()
                                .size(16.0),
                        );
                        let saved = stats.saved_percentage();
                        let saved_text = if saved >= 0.0 {
                            format!("{saved:.1}%")
                        } else {
                            format!("+{:.1}% larger", saved.abs())
                        };
                        ui.label(RichText::new(saved_text).strong().size(16.0));
                        ui.end_row();
                    });

                if over_2mb {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(
                            "Warning: Output still exceeds 2.0 MB. Try JPEG format with a lower Resolution Scale (e.g., 0.25x) or lower Quality (e.g., 35).",
                        )
                        .size(12.0)
                        .color(if dark {
                            Color32::from_rgb(255, 160, 160)
                        } else {
                            Color32::from_rgb(165, 29, 29)
                        }),
                    );
                } else if near_2mb {
                    ui.add_space(6.0);
                    ui.label(
                        RichText::new(
                            "Heads-up: Output is under 2.0 MiB (2,097,152 B) but slightly above 2,000,000 bytes. Some strict upload portals may reject it.",
                        )
                        .size(12.0),
                    );
                }

                ui.add_space(8.0);
                ui.label(
                    RichText::new(format!(
                        "Images: {} recompressed • {} kept original • {} small/masks skipped • {} unsupported skipped",
                        stats.images_recompressed,
                        stats.images_kept_original,
                        stats.images_skipped_small_or_mask,
                        stats.images_skipped_unsupported,
                    ))
                    .size(12.0)
                    .color(ui.visuals().weak_text_color()),
                );

                if let Some(inter_bytes) = stats.intermediate_format_bytes {
                    ui.label(
                        RichText::new(format!(
                            "Intermediate {} stream payload total: {} (transcoded to PDF-compatible streams)",
                            result.format_name,
                            format_bytes(inter_bytes)
                        ))
                        .size(11.5)
                        .color(ui.visuals().weak_text_color()),
                    );
                }

                ui.label(
                    RichText::new(format!("Saved to: {}", result.saved_path.display()))
                        .size(11.5)
                        .color(ui.visuals().weak_text_color()),
                );

                for reason in &stats.skipped_reasons {
                    ui.label(
                        RichText::new(format!("• {reason}"))
                            .size(11.5)
                            .color(ui.visuals().weak_text_color()),
                    );
                }
            });
    }
}

impl eframe::App for PdfCompressorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_worker();
        self.handle_drag_and_drop(&ctx);

        let busy = self.is_busy();
        let is_hovered = ctx.input(|i| !i.raw.hovered_files.is_empty());

        egui::CentralPanel::default().show(ui, |ui| {
            egui::ScrollArea::vertical().show(ui, |ui| {
                ui.add_space(4.0);
                ui.heading("PDF Image Compressor");
                ui.label(
                    RichText::new(
                        "Shrink scanned PDFs and photo documents while keeping exact page size and visual layout intact.",
                    )
                    .size(13.0)
                    .color(ui.visuals().weak_text_color()),
                );
                ui.add_space(12.0);

                self.render_file_card(ui, busy, is_hovered);
                ui.add_space(12.0);

                self.render_controls_card(ui, busy);
                ui.add_space(14.0);

                // Action bar: Compress & Save As...
                let can_compress = self.loaded.is_some() && !busy;
                ui.horizontal(|ui| {
                    let action_btn = egui::Button::new(
                        RichText::new("Compress & Save As...")
                            .size(15.5)
                            .strong()
                            .color(if can_compress {
                                Color32::WHITE
                            } else {
                                ui.visuals().weak_text_color()
                            }),
                    )
                    .fill(if can_compress {
                        Color32::from_rgb(37, 99, 235)
                    } else {
                        ui.visuals().widgets.inactive.bg_fill
                    })
                    .min_size(egui::vec2(210.0, 42.0));

                    if ui.add_enabled(can_compress, action_btn).clicked() {
                        self.start_compression(&ctx);
                    }

                    if busy {
                        ui.add_space(10.0);
                        ui.add(egui::Spinner::new());
                        if let Some(p) = &self.progress {
                            let frac = (p.current as f32) / (p.total.max(1) as f32);
                            ui.add(
                                egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                                    .text(&p.message)
                                    .show_percentage(),
                            );
                        }
                    }
                });

                if let Some(err) = &self.error_message {
                    ui.add_space(12.0);
                    let dark = ui.visuals().dark_mode;
                    Frame::new()
                        .fill(if dark {
                            Color32::from_rgb(56, 26, 26)
                        } else {
                            Color32::from_rgb(254, 236, 236)
                        })
                        .stroke(Stroke::new(1.0, Color32::from_rgb(217, 65, 65)))
                        .corner_radius(CornerRadius::same(8))
                        .inner_margin(Margin::same(10))
                        .show(ui, |ui| {
                            ui.set_width(ui.available_width());
                            ui.label(
                                RichText::new(format!("Error: {err}"))
                                    .strong()
                                    .color(if dark {
                                        Color32::from_rgb(255, 165, 165)
                                    } else {
                                        Color32::from_rgb(165, 29, 29)
                                    }),
                            );
                        });
                }

                if let Some(result) = &self.last_result {
                    ui.add_space(12.0);
                    self.render_summary_card(ui, result);
                } else if let Some(msg) = &self.status_message {
                    ui.add_space(10.0);
                    ui.label(
                        RichText::new(msg)
                            .size(12.5)
                            .color(ui.visuals().weak_text_color()),
                    );
                }
            });
        });
    }
}

fn format_bytes(bytes: usize) -> String {
    const KB: f64 = 1024.0;
    const MB: f64 = 1024.0 * 1024.0;
    let b = bytes as f64;
    if b >= MB {
        format!("{:.2} MB", b / MB)
    } else if b >= KB {
        format!("{:.1} KB", b / KB)
    } else {
        format!("{bytes} B")
    }
}
