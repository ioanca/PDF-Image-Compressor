use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver};
use std::sync::Arc;
use std::thread;

use eframe::egui::{self, Color32, CornerRadius, Frame, Margin, RichText, Stroke};
use rfd::FileDialog;

use crate::pdf_processor::{
    bytes_to_mb, compress_pdf, inspect_pdf, mb_to_bytes, CompressionOptions, CompressionStats,
    ImageFormatChoice, PdfSummary, ProgressUpdate, ResizeFilterChoice,
};

pub const DEFAULT_WINDOW_WIDTH: f32 = 520.0;
pub const DEFAULT_WINDOW_HEIGHT: f32 = 440.0;
pub const MIN_WINDOW_WIDTH: f32 = 480.0;
pub const MIN_WINDOW_HEIGHT: f32 = 420.0;

struct LoadedPdf {
    path: Option<PathBuf>,
    file_name: String,
    bytes: Arc<Vec<u8>>,
    summary: PdfSummary,
}

enum WorkerEvent {
    Progress(ProgressUpdate),
    Finished(Result<(Vec<u8>, CompressionStats), String>),
}

struct PendingSave {
    bytes: Vec<u8>,
    stats: CompressionStats,
    default_file_name: String,
    initial_dir: Option<PathBuf>,
}

struct CompletedResult {
    saved_path: PathBuf,
    stats: CompressionStats,
}

pub struct PdfCompressorApp {
    loaded: Option<LoadedPdf>,
    options: CompressionOptions,
    target_size_enabled: bool,
    target_size_mb: f32,
    worker_rx: Option<Receiver<WorkerEvent>>,
    progress: Option<ProgressUpdate>,
    last_result: Option<CompletedResult>,
    status_message: Option<String>,
    error_message: Option<String>,
    last_fitted_height: Option<f32>,
    #[cfg(test)]
    last_content_width: f32,
}

impl Default for PdfCompressorApp {
    fn default() -> Self {
        Self {
            loaded: None,
            options: CompressionOptions::default(),
            target_size_enabled: false,
            target_size_mb: 2.0,
            worker_rx: None,
            progress: None,
            last_result: None,
            status_message: None,
            error_message: None,
            last_fitted_height: None,
            #[cfg(test)]
            last_content_width: 0.0,
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
                let original_mb = bytes_to_mb(summary.file_bytes) as f32;
                self.target_size_mb = self.target_size_mb.clamp(0.0, original_mb);
                if self.target_size_enabled && self.target_size_mb > 0.0 {
                    let target_bytes = mb_to_bytes(f64::from(self.target_size_mb));
                    if target_bytes > 0 {
                        summary.apply_target_scale_first(&mut self.options, target_bytes);
                    }
                }
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
        thread::spawn(move || {
            let result = compress_pdf(&bytes, &options, |update| {
                let _ = tx.send(WorkerEvent::Progress(update));
                repaint_ctx.request_repaint();
            })
            .map_err(|e| e.to_string());

            let _ = tx.send(WorkerEvent::Finished(result));
            repaint_ctx.request_repaint();
        });
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker_rx else {
            return;
        };

        let mut finished_payload: Option<Result<(Vec<u8>, CompressionStats), String>> = None;
        while let Ok(event) = rx.try_recv() {
            match event {
                WorkerEvent::Progress(p) => {
                    self.progress = Some(p);
                }
                WorkerEvent::Finished(result) => {
                    finished_payload = Some(result);
                }
            }
        }

        if let Some(result) = finished_payload {
            self.worker_rx = None;
            self.progress = None;

            match result {
                Ok((out_bytes, stats)) => {
                    let pending = self.build_pending_save(out_bytes, stats);
                    self.prompt_save_as(pending);
                }
                Err(err) => {
                    self.error_message = Some(err);
                }
            }
        }
    }

    fn build_pending_save(&self, bytes: Vec<u8>, stats: CompressionStats) -> PendingSave {
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
        } else {
            neutral_card_fill(dark)
        };
        let border_color = if is_hovered {
            Color32::from_rgb(66, 133, 244)
        } else {
            neutral_card_border(dark)
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
                        RichText::new("Select PDF").size(14.5).strong(),
                    )
                    .min_size(egui::vec2(112.0, 38.0));

                    if ui.add_enabled(!busy, select_btn).clicked() {
                        self.open_file_dialog();
                    }

                    ui.add_space(10.0);

                    const COMPRESS_BTN_WIDTH: f32 = 152.0;
                    let middle_w = (ui.available_width() - COMPRESS_BTN_WIDTH - 12.0).max(80.0);
                    let middle_layout = if self.loaded.is_some() {
                        egui::Layout::top_down(egui::Align::Min)
                    } else {
                        egui::Layout::left_to_right(egui::Align::Center)
                    };
                    ui.allocate_ui_with_layout(
                        egui::vec2(middle_w, 38.0),
                        middle_layout,
                        |ui| {
                            if let Some(loaded) = &self.loaded {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(&loaded.file_name).strong().size(14.5),
                                    )
                                    .truncate(),
                                );
                                ui.add(
                                    egui::Label::new(
                                        RichText::new(format!(
                                            "{} page(s) • {} image(s)",
                                            loaded.summary.page_count, loaded.summary.image_count
                                        ))
                                        .size(12.5)
                                        .color(ui.visuals().weak_text_color()),
                                    )
                                    .truncate(),
                                );
                            } else {
                                ui.add(
                                    egui::Label::new(
                                        RichText::new("No PDF selected").strong().size(14.5),
                                    )
                                    .truncate(),
                                );
                            }
                        },
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_compress = self.loaded.is_some() && !busy;
                        let action_btn = egui::Button::new(
                            RichText::new("Compress & Save")
                                .size(14.5)
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
                        .min_size(egui::vec2(COMPRESS_BTN_WIDTH, 38.0));

                        if ui.add_enabled(can_compress, action_btn).clicked() {
                            let ctx = ui.ctx().clone();
                            self.start_compression(&ctx);
                        }
                    });
                });
            });
    }

    fn render_controls_card(&mut self, ui: &mut egui::Ui, busy: bool) {
        let dark = ui.visuals().dark_mode;
        let card_fill = neutral_card_fill(dark);
        let border_color = neutral_card_border(dark);

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
                    let mut format_changed = false;
                    let mut quality_changed = false;
                    let mut scale_changed = false;
                    let mut filter_changed = false;
                    let mut target_changed = false;
                    let mut enabled_toggled_on = false;

                    // Format / Algorithm dropdown
                    let prev_format = self.options.format;
                    ui.horizontal(|ui| {
                        ui.label("Image Format:");
                        egui::ComboBox::from_id_salt("image_format_selector")
                            .selected_text(self.options.format.label())
                            .width(90.0)
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
                    if self.options.format != prev_format {
                        format_changed = true;
                    }

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
                    let prev_quality = self.options.quality;
                    ui.horizontal(|ui| {
                        ui.label("Quality:");
                        let slider = egui::Slider::new(&mut self.options.quality, 1..=100)
                            .clamping(egui::SliderClamping::Always);
                        let quality_resp = ui.add_enabled(uses_quality, slider);
                        if uses_quality
                            && quality_resp.changed()
                            && self.options.quality != prev_quality
                        {
                            quality_changed = true;
                        }
                        if !uses_quality {
                            ui.label(
                                RichText::new("(Lossless mode — ignored)")
                                    .size(12.0)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        }
                    });

                    ui.add_space(8.0);

                    // Resolution Scale slider
                    let prev_scale = self.options.scale;
                    ui.horizontal(|ui| {
                        ui.label("Resolution Scale:");
                        let scale_resp = ui.add(
                            egui::Slider::new(&mut self.options.scale, 0.10..=1.00)
                                .step_by(0.05)
                                .clamping(egui::SliderClamping::Edits)
                                .fixed_decimals(2)
                                .suffix("x"),
                        );
                        if scale_resp.changed() && (self.options.scale - prev_scale).abs() >= 1e-4
                        {
                            scale_changed = true;
                        }
                    });

                    ui.add_space(8.0);

                    // Resize filter choice
                    let prev_filter = self.options.resize_filter;
                    ui.horizontal(|ui| {
                        ui.label("Resize Filter:");
                        for filter in ResizeFilterChoice::ALL {
                            let selected = self.options.resize_filter == filter;
                            if ui.selectable_label(selected, filter.label()).clicked() {
                                self.options.resize_filter = filter;
                            }
                        }
                    });
                    if self.options.resize_filter != prev_filter {
                        filter_changed = true;
                    }

                    ui.add_space(8.0);

                    // Unified toggleable Target Size / Estimated Size row
                    ui.horizontal(|ui| {
                        let prev_enabled = self.target_size_enabled;
                        let cb_label = if self.target_size_enabled {
                            "Target Size (MB):"
                        } else {
                            "Target Size"
                        };
                        let cb_resp = ui.checkbox(&mut self.target_size_enabled, cb_label);
                        if cb_resp.changed() && self.target_size_enabled && !prev_enabled {
                            enabled_toggled_on = true;
                        }

                        if !self.target_size_enabled {
                            ui.add_space(6.0);
                            self.render_estimated_size_inline(ui);
                        } else {
                            let has_file = self.loaded.is_some();
                            let max_target_mb = self
                                .loaded
                                .as_ref()
                                .map(|l| (bytes_to_mb(l.summary.file_bytes) as f32).max(1e-6))
                                .unwrap_or(10.0);
                            let target_decimals = if has_file {
                                target_decimals_for_mb(f64::from(max_target_mb))
                            } else {
                                2
                            };
                            let prev_target = self.target_size_mb;

                            ui.add_enabled_ui(has_file, |ui| {
                                let slider_resp = ui.add(
                                    egui::Slider::new(
                                        &mut self.target_size_mb,
                                        0.0..=max_target_mb,
                                    )
                                    .show_value(false)
                                    .clamping(egui::SliderClamping::Always),
                                );
                                let drag_resp = ui.add(
                                    egui::DragValue::new(&mut self.target_size_mb)
                                        .range(0.0..=max_target_mb)
                                        .speed((f64::from(max_target_mb) / 200.0).clamp(0.001, 0.25))
                                        .fixed_decimals(target_decimals)
                                        .suffix(" MB"),
                                );
                                if (slider_resp.changed() || drag_resp.changed())
                                    && (self.target_size_mb - prev_target).abs() >= 1e-5
                                {
                                    target_changed = true;
                                }
                            });

                            if has_file {
                                self.target_size_mb = self.target_size_mb.clamp(0.0, max_target_mb);
                                if self.target_size_mb <= 0.0 {
                                    ui.label(
                                        RichText::new("(No limit)")
                                            .size(12.0)
                                            .color(ui.visuals().weak_text_color()),
                                    );
                                }
                            }
                        }
                    });

                    // Apply automatic inverse coupling strictly on user-changed events to avoid loops.
                    if self.target_size_enabled && self.target_size_mb > 0.0 {
                        if let Some(loaded) = &self.loaded {
                            let target_bytes = mb_to_bytes(f64::from(self.target_size_mb));
                            if target_bytes > 0 {
                                let before_q = self.options.quality;
                                let before_s = self.options.scale;

                                if enabled_toggled_on
                                    || target_changed
                                    || format_changed
                                    || filter_changed
                                {
                                    loaded
                                        .summary
                                        .apply_target_scale_first(&mut self.options, target_bytes);
                                } else if quality_changed {
                                    self.options.scale = loaded
                                        .summary
                                        .solve_scale_for_target(&self.options, target_bytes);
                                } else if scale_changed && self.options.format.uses_quality() {
                                    self.options.quality = loaded
                                        .summary
                                        .solve_quality_for_target(&self.options, target_bytes);
                                }

                                if self.options.quality != before_q
                                    || (self.options.scale - before_s).abs() >= 1e-4
                                {
                                    ui.ctx().request_repaint();
                                }
                            }
                        }
                    }
                });
            });
    }

    fn render_estimated_size_inline(&self, ui: &mut egui::Ui) {
        let Some(loaded) = &self.loaded else {
            ui.label(
                RichText::new("Estimated Size: —")
                    .size(12.5)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        ui.label(
            RichText::new(estimated_size_text(loaded, &self.options))
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
    }

    fn summary_metrics(&self) -> (String, String, String, bool) {
        match &self.last_result {
            Some(result) => {
                let stats = &result.stats;
                let saved = stats.saved_percentage();
                let saved_str = if saved >= 0.0 {
                    format!("{saved:.1}%")
                } else {
                    format!("+{:.1}% larger", saved.abs())
                };
                (
                    format_bytes(stats.original_bytes),
                    format_bytes(stats.compressed_bytes),
                    saved_str,
                    false,
                )
            }
            None => match &self.loaded {
                Some(loaded) => {
                    let orig_bytes = loaded.summary.file_bytes;
                    let (saved_str, is_est) = if orig_bytes > 0 {
                        let est_bytes = loaded.summary.estimate_compressed_bytes(&self.options);
                        let saved_pct =
                            ((orig_bytes as f64 - est_bytes as f64) / orig_bytes as f64) * 100.0;
                        let s = if saved_pct >= 0.0 {
                            format!("~{saved_pct:.0}%")
                        } else {
                            format!("~+{:.0}%", saved_pct.abs())
                        };
                        (s, true)
                    } else {
                        ("-".to_string(), false)
                    };
                    (format_bytes(orig_bytes), "-".to_string(), saved_str, is_est)
                }
                None => ("-".to_string(), "-".to_string(), "-".to_string(), false),
            },
        }
    }

    fn render_summary_card(&self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        let target_bytes = mb_to_bytes(f64::from(self.target_size_mb));
        let has_active_target =
            self.target_size_enabled && self.target_size_mb > 0.0 && target_bytes > 0;
        let over_target = self.last_result.as_ref().is_some_and(|result| {
            has_active_target && result.stats.exceeds_target_mb(f64::from(self.target_size_mb))
        });

        let (card_fill, border_color) = match &self.last_result {
            None => (neutral_card_fill(dark), neutral_card_border(dark)),
            Some(_) if over_target => {
                if dark {
                    (Color32::from_rgb(48, 30, 30), Color32::from_rgb(200, 75, 75))
                } else {
                    (Color32::from_rgb(255, 243, 243), Color32::from_rgb(217, 83, 79))
                }
            }
            Some(_) => {
                if dark {
                    (Color32::from_rgb(26, 44, 34), Color32::from_rgb(60, 160, 95))
                } else {
                    (Color32::from_rgb(240, 252, 244), Color32::from_rgb(52, 168, 83))
                }
            }
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
                    if self.last_result.is_some() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            let (badge_text, badge_fill) = if over_target {
                                ("OVER TARGET", Color32::from_rgb(198, 40, 40))
                            } else if has_active_target {
                                ("UNDER TARGET", Color32::from_rgb(34, 139, 72))
                            } else {
                                ("COMPRESSED", Color32::from_rgb(34, 139, 72))
                            };

                            Frame::new()
                                .fill(badge_fill)
                                .corner_radius(CornerRadius::same(6))
                                .inner_margin(Margin::symmetric(8, 4))
                                .show(ui, |ui| {
                                    ui.label(
                                        RichText::new(badge_text)
                                            .strong()
                                            .size(12.0)
                                            .color(Color32::WHITE),
                                    );
                                });
                        });
                    }
                });

                ui.add_space(8.0);

                let (orig_text, new_text, saved_text, saved_is_estimate) = self.summary_metrics();

                egui::Grid::new("summary_metrics_grid")
                    .num_columns(3)
                    .min_col_width(110.0)
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

                        ui.label(RichText::new(orig_text).strong().size(16.0));
                        ui.label(RichText::new(new_text).strong().size(16.0));
                        let saved_rich = if saved_is_estimate {
                            RichText::new(saved_text)
                                .size(16.0)
                                .color(ui.visuals().weak_text_color())
                        } else {
                            RichText::new(saved_text).strong().size(16.0)
                        };
                        ui.label(saved_rich);
                        ui.end_row();
                    });

                if let Some(result) = &self.last_result {
                    if over_target {
                        let excess = result.stats.compressed_bytes.saturating_sub(target_bytes);
                        let target_decimals = self
                            .loaded
                            .as_ref()
                            .map(|l| target_decimals_for_mb(bytes_to_mb(l.summary.file_bytes)))
                            .unwrap_or(2);
                        ui.add_space(6.0);
                        ui.label(
                            RichText::new(format!(
                                "Warning: Output ({}) exceeds your {:.prec$} MB target by {}. Try JPEG format with a lower Resolution Scale or lower Quality.",
                                format_bytes(result.stats.compressed_bytes),
                                self.target_size_mb,
                                format_bytes(excess),
                                prec = target_decimals,
                            ))
                            .size(12.0)
                            .color(if dark {
                                Color32::from_rgb(255, 160, 160)
                            } else {
                                Color32::from_rgb(165, 29, 29)
                            }),
                        );
                    }

                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("Saved to: {}", result.saved_path.display()))
                            .size(11.5)
                            .color(ui.visuals().weak_text_color()),
                    );
                }
            });
    }
    fn render_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_worker();
        self.handle_drag_and_drop(&ctx);

        let busy = self.is_busy();
        let is_hovered = ctx.input(|i| !i.raw.hovered_files.is_empty());
        let fallback_size = ui.max_rect().size();

        let panel_frame = Frame::central_panel(ui.style()).inner_margin(Margin::same(10));
        let panel_v_margin =
            f32::from(panel_frame.inner_margin.top) + f32::from(panel_frame.inner_margin.bottom);

        let panel_resp = egui::CentralPanel::default()
            .frame(panel_frame)
            .show(ui, |ui| {
                egui::ScrollArea::vertical().show(ui, |ui| {
                    self.render_file_card(ui, busy, is_hovered);
                    ui.add_space(10.0);

                    self.render_controls_card(ui, busy);

                    if busy {
                        ui.add_space(10.0);
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new());
                            if let Some(p) = &self.progress {
                                let frac = (p.current as f32) / (p.total.max(1) as f32);
                                ui.add(
                                    egui::ProgressBar::new(frac.clamp(0.0, 1.0))
                                        .text(&p.message)
                                        .show_percentage(),
                                );
                            }
                        });
                    }

                    if let Some(err) = &self.error_message {
                        ui.add_space(10.0);
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

                    if self.last_result.is_none() {
                        if let Some(msg) = &self.status_message {
                            ui.add_space(10.0);
                            ui.label(
                                RichText::new(msg)
                                    .size(12.5)
                                    .color(ui.visuals().weak_text_color()),
                            );
                        }
                    }

                    ui.add_space(10.0);
                    self.render_summary_card(ui);
                })
            });

        // Fit the window height cleanly to the rendered cards + panel margins whenever
        // layout height changes (initial draw, "Saved to: ..." appearing, notes, etc.).
        #[cfg(test)]
        {
            self.last_content_width = panel_resp.inner.content_size.x;
        }
        let content_height = panel_resp.inner.content_size.y;
        if content_height.is_finite() && content_height > 0.0 {
            let ideal_height = (content_height + panel_v_margin + 2.0).ceil();
            let height_changed = self
                .last_fitted_height
                .is_none_or(|prev| (prev - ideal_height).abs() > 1.0);
            if height_changed {
                self.last_fitted_height = Some(ideal_height);
                let target_min_h = ideal_height.min(MIN_WINDOW_HEIGHT);
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                    MIN_WINDOW_WIDTH,
                    target_min_h,
                )));
                let current_size = ctx
                    .input(|i| i.viewport().inner_rect.map(|r| r.size()))
                    .unwrap_or(fallback_size);
                if (current_size.y - ideal_height).abs() > 1.0 {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        current_size.x.max(MIN_WINDOW_WIDTH),
                        ideal_height.max(target_min_h),
                    )));
                }
            }
        }
    }
}

impl eframe::App for PdfCompressorApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        self.render_ui(ui);
    }
}

fn estimated_size_text(loaded: &LoadedPdf, options: &CompressionOptions) -> String {
    let est_bytes = loaded.summary.estimate_compressed_bytes(options);
    format!("Estimated Size: {}", format_bytes(est_bytes))
}

fn neutral_card_fill(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(30, 33, 40)
    } else {
        Color32::from_rgb(246, 248, 251)
    }
}

fn neutral_card_border(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgb(56, 62, 74)
    } else {
        Color32::from_rgb(214, 220, 229)
    }
}

fn target_decimals_for_mb(max_mb: f64) -> usize {
    if max_mb < 0.005 {
        4
    } else if max_mb < 0.05 {
        3
    } else {
        2
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run_frame(ctx: &egui::Context, app: &mut PdfCompressorApp, width: f32, height: f32) -> egui::FullOutput {
        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, height),
            )),
            ..Default::default()
        };
        if let Some(vp) = raw_input.viewports.get_mut(&egui::ViewportId::ROOT) {
            vp.inner_rect = Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, height),
            ));
        }
        let mut out = ctx.run_ui(raw_input, |ui| app.render_ui(ui));
        out.textures_delta.clear();
        out
    }

    fn has_inner_size_cmd(output: &egui::FullOutput) -> Option<egui::Vec2> {
        output
            .viewport_output
            .get(&egui::ViewportId::ROOT)
            .and_then(|vo| {
                vo.commands.iter().rev().find_map(|cmd| match cmd {
                    egui::ViewportCommand::InnerSize(sz) => Some(*sz),
                    _ => None,
                })
            })
    }

    #[test]
    fn tight_viewport_height_fits_before_and_after_save_without_churn() {
        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();

        // Initial launch at 520x440
        let out1 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        let initial_fitted = app
            .last_fitted_height
            .expect("initial height should be measured");
        assert!(
            (370.0..=465.0).contains(&initial_fitted),
            "expected tight initial height before compression, got {initial_fitted}"
        );

        // Next frame at fitted height emits no further InnerSize command (no jitter/loop)
        let out2 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, initial_fitted);
        assert_eq!(has_inner_size_cmd(&out2), None);
        let _ = out1;

        // Simulate completed save ("Saved to: ..." appears in Compression Summary)
        app.last_result = Some(CompletedResult {
            saved_path: PathBuf::from(r"C:\Users\demo\homework_compressed.pdf"),
            stats: CompressionStats {
                original_bytes: 4 * 1024 * 1024,
                compressed_bytes: 1_200_000,
                ..Default::default()
            },
        });

        let out3 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, initial_fitted);
        let post_save_fitted = app.last_fitted_height.expect("post-save height measured");
        assert!(
            post_save_fitted > initial_fitted,
            "window should expand for 'Saved to: ...' ({initial_fitted} -> {post_save_fitted})"
        );
        let requested = has_inner_size_cmd(&out3).expect("should emit InnerSize when height grows");
        assert!((requested.x - DEFAULT_WINDOW_WIDTH).abs() < 0.5);
        assert!((requested.y - post_save_fitted).abs() < 0.5);

        // Subsequent frame at new fitted height settles with no further resize
        let out4 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, post_save_fitted);
        assert_eq!(has_inner_size_cmd(&out4), None);

        // Narrowest supported width (480px) fits controls cleanly without horizontal overflow or wrapping taller
        let mut compact_app = PdfCompressorApp::default();
        compact_app.loaded = Some(LoadedPdf {
            path: None,
            file_name: "homework_scan_2026.pdf".to_string(),
            bytes: Arc::new(Vec::new()),
            summary: PdfSummary {
                page_count: 5,
                image_count: 5,
                compressible_image_count: 5,
                file_bytes: 3_500_000,
                non_image_overhead_bytes: 12_000,
                compressible_rgb_pixels: 5_000_000,
                compressible_gray_pixels: 0,
                orig_dct_image_count: 5,
                orig_dct_bytes: 3_488_000,
                orig_dct_rgb_pixels: 5_000_000,
                orig_dct_gray_pixels: 0,
                sample_rgb_jpeg_q50_bpp: 1.5,
                sample_rgb_flate_bpp: 10.0,
            },
        });
        compact_app.options.format = ImageFormatChoice::Png;
        compact_app.target_size_enabled = true;
        let _out5 = run_frame(&ctx, &mut compact_app, MIN_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        let compact_height = compact_app
            .last_fitted_height
            .expect("compact height should be measured");
        assert!(
            (370.0..=465.0).contains(&compact_height),
            "expected compact layout without extra line-wraps at {MIN_WINDOW_WIDTH}px width, got {compact_height}"
        );
        assert!(
            compact_app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "expected controls at {MIN_WINDOW_WIDTH}px window width to fit inside {}px panel width without horizontal overflow, got {}px",
            MIN_WINDOW_WIDTH - 20.0,
            compact_app.last_content_width
        );
    }

    #[test]
    fn estimated_size_and_summary_metrics_follow_state_transitions() {
        let mut app = PdfCompressorApp::default();

        // Unloaded: summary shows dashes
        assert_eq!(
            app.summary_metrics(),
            ("-".to_string(), "-".to_string(), "-".to_string(), false)
        );

        // Loaded before compression: inline estimate has no tilde and summary shows muted estimate
        let loaded = LoadedPdf {
            path: None,
            file_name: "notes.pdf".to_string(),
            bytes: Arc::new(Vec::new()),
            summary: PdfSummary {
                page_count: 2,
                image_count: 2,
                compressible_image_count: 2,
                file_bytes: 4_000_000,
                non_image_overhead_bytes: 10_000,
                compressible_rgb_pixels: 4_000_000,
                compressible_gray_pixels: 0,
                orig_dct_image_count: 2,
                orig_dct_bytes: 3_990_000,
                orig_dct_rgb_pixels: 4_000_000,
                orig_dct_gray_pixels: 0,
                sample_rgb_jpeg_q50_bpp: 1.5,
                sample_rgb_flate_bpp: 10.0,
            },
        };
        let inline_est = estimated_size_text(&loaded, &app.options);
        assert!(inline_est.starts_with("Estimated Size: "));
        assert!(!inline_est.contains('~'));
        assert!(!inline_est.contains('%'));

        app.loaded = Some(loaded);
        let (orig, new_sz, saved, is_est) = app.summary_metrics();
        assert_ne!(orig, "-");
        assert_eq!(new_sz, "-");
        assert!(saved.starts_with('~') && saved.ends_with('%') && !saved.contains("est."));
        assert!(is_est);
        assert_eq!(ImageFormatChoice::Png.label(), "PNG");

        // After compression: actual percentage replaces muted estimate as bold actual text
        app.last_result = Some(CompletedResult {
            saved_path: PathBuf::from("notes_compressed.pdf"),
            stats: CompressionStats {
                original_bytes: 4_000_000,
                compressed_bytes: 1_156_000,
                ..Default::default()
            },
        });
        let (orig, new_sz, saved, is_est) = app.summary_metrics();
        assert_ne!(orig, "-");
        assert_ne!(new_sz, "-");
        assert_eq!(saved, "71.1%");
        assert!(!is_est);
    }
}
