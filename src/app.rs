use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
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

const FILE_TABLE_MAX_VISIBLE_ROWS: usize = 10;
const SUMMARY_TABLE_MAX_VISIBLE_ROWS: usize = 6;
const TABLE_HEADER_HEIGHT: f32 = 24.0;
const TABLE_ROW_HEIGHT: f32 = 24.0;
const TABLE_ROW_GAP: f32 = 2.0;
const TABLE_NO_COL_WIDTH: f32 = 44.0;
const FILE_TABLE_NO_COL_WIDTH: f32 = 40.0;
const FILE_TABLE_SIZE_COL_WIDTH: f32 = 85.0;
const FILE_TABLE_EST_COL_WIDTH: f32 = 85.0;
const TABLE_CELL_PAD_X: f32 = 7.0;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
enum TargetMode {
    #[default]
    Off,
    Megabytes,
    Percent,
}

impl TargetMode {
    fn next(self) -> Self {
        match self {
            Self::Off => Self::Megabytes,
            Self::Megabytes => Self::Percent,
            Self::Percent => Self::Off,
        }
    }

    fn is_enabled(self) -> bool {
        !matches!(self, Self::Off)
    }

    fn label(self) -> &'static str {
        match self {
            Self::Off => "Target Size",
            Self::Megabytes => "Target Size (MB):",
            Self::Percent => "Target Size (%):",
        }
    }
}

#[derive(Clone, PartialEq)]
struct CompressionSettings {
    options: CompressionOptions,
    target_mode: TargetMode,
    target_size_mb: f32,
    target_size_pct: f32,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            options: CompressionOptions::default(),
            target_mode: TargetMode::Off,
            target_size_mb: 10.0,
            target_size_pct: 100.0,
        }
    }
}

impl CompressionSettings {
    fn for_file(original_mb: f32) -> Self {
        Self {
            target_size_mb: original_mb.max(0.0),
            target_size_pct: 100.0,
            ..Self::default()
        }
    }

    fn clamped_for_file(&self, original_mb: f32) -> Self {
        let mut cloned = self.clone();
        cloned.target_size_mb = cloned.target_size_mb.clamp(0.0, original_mb.max(0.0));
        cloned.target_size_pct = cloned.target_size_pct.clamp(1.0, 100.0);
        cloned
    }

    fn effective_target_mb(&self, original_size_mb: f64) -> Option<f64> {
        match self.target_mode {
            TargetMode::Off => None,
            TargetMode::Megabytes => {
                if self.target_size_mb > 0.0 {
                    let orig_f32 = original_size_mb as f32;
                    if original_size_mb > 0.0
                        && (self.target_size_mb >= orig_f32
                            || (orig_f32 - self.target_size_mb).abs()
                                <= f32::EPSILON * orig_f32.max(1.0))
                    {
                        Some(original_size_mb)
                    } else {
                        Some(f64::from(self.target_size_mb))
                    }
                } else {
                    None
                }
            }
            TargetMode::Percent => {
                if original_size_mb > 0.0 {
                    let pct = f64::from(self.target_size_pct.clamp(1.0, 100.0));
                    if pct >= 100.0 {
                        Some(original_size_mb)
                    } else {
                        Some(original_size_mb * (pct / 100.0))
                    }
                } else {
                    None
                }
            }
        }
    }
}

#[derive(Clone, Debug)]
struct CompressionResult {
    saved_path: PathBuf,
    stats: CompressionStats,
    new_size_mb: f64,
    space_saved_pct: f64,
    over_target: bool,
}

impl CompressionResult {
    fn from_stats(
        saved_path: PathBuf,
        stats: CompressionStats,
        settings: &CompressionSettings,
        original_size_mb: f64,
    ) -> Self {
        let new_size_mb = bytes_to_mb(stats.compressed_bytes);
        let space_saved_pct = stats.saved_percentage();
        let over_target = settings
            .effective_target_mb(original_size_mb)
            .is_some_and(|mb| mb > 0.0 && stats.exceeds_target_mb(mb));
        Self {
            saved_path,
            stats,
            new_size_mb,
            space_saved_pct,
            over_target,
        }
    }
}

struct PdfFileItem {
    path: Option<PathBuf>,
    file_name: String,
    original_size_mb: f64,
    bytes: Arc<Vec<u8>>,
    summary: PdfSummary,
    settings: CompressionSettings,
    cached_estimated_bytes: usize,
    result: Option<CompressionResult>,
    error: Option<String>,
}

impl PdfFileItem {
    fn estimated_compressed_bytes(&self) -> usize {
        self.cached_estimated_bytes
    }

    fn estimated_saved_percentage(&self) -> Option<f64> {
        let orig = self.summary.file_bytes;
        if orig == 0 {
            return None;
        }
        let est = self.estimated_compressed_bytes();
        Some(((orig as f64 - est as f64) / orig as f64) * 100.0)
    }
}

struct BatchJob {
    index: usize,
    original_size_mb: f64,
    bytes: Arc<Vec<u8>>,
    settings: CompressionSettings,
    output_path: PathBuf,
}

enum WorkerEvent {
    Progress(ProgressUpdate),
    LoadFinished {
        items: Vec<PdfFileItem>,
        errors: Vec<String>,
    },
    SingleFinished(Result<(Vec<u8>, CompressionStats), String>),
    BatchItemFinished {
        index: usize,
        outcome: Result<CompressionResult, String>,
    },
    BatchFinished {
        folder: PathBuf,
    },
}

struct PendingSave {
    bytes: Vec<u8>,
    stats: CompressionStats,
    default_file_name: String,
    initial_dir: Option<PathBuf>,
}

#[derive(Default, Clone, Copy)]
struct SettingsDelta {
    format_changed: bool,
    quality_changed: bool,
    scale_changed: bool,
    filter_changed: bool,
    target_mode_changed: bool,
    target_mode_activated: bool,
    target_mb_changed: bool,
    target_pct_changed: bool,
}

impl SettingsDelta {
    fn any_changed(self) -> bool {
        self.format_changed
            || self.quality_changed
            || self.scale_changed
            || self.filter_changed
            || self.target_mode_changed
            || self.target_mb_changed
            || self.target_pct_changed
    }
}

pub struct PdfCompressorApp {
    files: Vec<PdfFileItem>,
    selected_indices: BTreeSet<usize>,
    selection_anchor: Option<usize>,
    apply_for_all: bool,
    est_show_percent: bool,
    /// Keeps the user's shared Target Size slider value when `Apply for ALL` is checked
    /// even if smaller individual files clamp their own per-file `target_size_mb`.
    shared_target_size_mb: Option<f32>,
    unloaded_settings: CompressionSettings,
    worker_rx: Option<Receiver<WorkerEvent>>,
    progress: Option<ProgressUpdate>,
    batch_saved_folder: Option<PathBuf>,
    status_message: Option<String>,
    error_message: Option<String>,
    last_fitted_height: Option<f32>,
    #[cfg(test)]
    last_content_width: f32,
    #[cfg(test)]
    last_quality_slider_rect: Option<egui::Rect>,
    #[cfg(test)]
    last_scale_slider_rect: Option<egui::Rect>,
    #[cfg(test)]
    last_quality_drag_rect: Option<egui::Rect>,
    #[cfg(test)]
    last_scale_drag_rect: Option<egui::Rect>,
}

impl Default for PdfCompressorApp {
    fn default() -> Self {
        Self {
            files: Vec::new(),
            selected_indices: BTreeSet::new(),
            selection_anchor: None,
            apply_for_all: false,
            est_show_percent: false,
            shared_target_size_mb: None,
            unloaded_settings: CompressionSettings::default(),
            worker_rx: None,
            progress: None,
            batch_saved_folder: None,
            status_message: None,
            error_message: None,
            last_fitted_height: None,
            #[cfg(test)]
            last_content_width: 0.0,
            #[cfg(test)]
            last_quality_slider_rect: None,
            #[cfg(test)]
            last_scale_slider_rect: None,
            #[cfg(test)]
            last_quality_drag_rect: None,
            #[cfg(test)]
            last_scale_drag_rect: None,
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

    fn is_multi_file(&self) -> bool {
        self.files.len() > 1
    }

    fn active_index(&self) -> Option<usize> {
        if self.files.is_empty() {
            return None;
        }
        if let Some(anchor) = self.selection_anchor {
            if anchor < self.files.len() && self.selected_indices.contains(&anchor) {
                return Some(anchor);
            }
        }
        if let Some(first) = self
            .selected_indices
            .iter()
            .copied()
            .find(|&i| i < self.files.len())
        {
            return Some(first);
        }
        if let Some(anchor) = self.selection_anchor {
            if anchor < self.files.len() {
                return Some(anchor);
            }
        }
        Some(0)
    }

    fn open_file_dialog(&mut self, ctx: &egui::Context) {
        let mut dialog = FileDialog::new().add_filter("PDF Document", &["pdf"]);
        if let Some(dir) = self
            .active_index()
            .and_then(|idx| self.files.get(idx))
            .and_then(|f| f.path.as_ref())
            .and_then(|p| p.parent())
        {
            dialog = dialog.set_directory(dir);
        }

        if let Some(paths) = dialog.pick_files() {
            if !paths.is_empty() {
                self.start_loading_job(paths, Vec::new(), Vec::new(), ctx);
            }
        }
    }

    fn inspect_loaded_inputs(
        inputs: Vec<(Option<PathBuf>, String, Vec<u8>)>,
        mut errors: Vec<String>,
        on_item: &(dyn Fn() + Sync),
    ) -> (Vec<PdfFileItem>, Vec<String>) {
        let inspected: Vec<(
            Option<PathBuf>,
            String,
            Vec<u8>,
            Result<PdfSummary, crate::pdf_processor::PdfError>,
        )> = inputs
            .into_iter()
            .map(|(path, file_name, bytes)| {
                let summary_res = inspect_pdf(&bytes);
                on_item();
                (path, file_name, bytes, summary_res)
            })
            .collect();

        let mut loaded_items = Vec::with_capacity(inspected.len());
        for (path, file_name, bytes, summary_res) in inspected {
            match summary_res {
                Ok(summary) => {
                    let original_size_mb = bytes_to_mb(summary.file_bytes);
                    let settings = CompressionSettings::for_file(original_size_mb as f32);
                    let cached_estimated_bytes = summary.estimate_compressed_bytes(&settings.options);
                    loaded_items.push(PdfFileItem {
                        path,
                        file_name,
                        original_size_mb,
                        bytes: Arc::new(bytes),
                        summary,
                        settings,
                        cached_estimated_bytes,
                        result: None,
                        error: None,
                    });
                }
                Err(err) => {
                    errors.push(format!("{file_name}: {err}"));
                }
            }
        }

        (loaded_items, errors)
    }

    fn apply_loaded_items(&mut self, loaded_items: Vec<PdfFileItem>, errors: Vec<String>) {
        if !loaded_items.is_empty() {
            self.files = loaded_items;
            self.selected_indices.clear();
            self.selected_indices.insert(0);
            self.selection_anchor = Some(0);
            self.batch_saved_folder = None;
            self.status_message = None;
            self.shared_target_size_mb = None;
            self.unloaded_settings = CompressionSettings::default();
        }

        self.error_message = if errors.is_empty() {
            None
        } else {
            Some(errors.join("; "))
        };
    }

    fn start_loading_job(
        &mut self,
        paths: Vec<PathBuf>,
        mut inputs: Vec<(Option<PathBuf>, String, Vec<u8>)>,
        mut errors: Vec<String>,
        ctx: &egui::Context,
    ) {
        let total = paths.len().saturating_add(inputs.len()).max(1);
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.error_message = None;
        self.status_message = None;
        self.progress = Some(ProgressUpdate {
            current: 0,
            total,
        });

        let repaint_ctx = ctx.clone();
        thread::spawn(move || {
            for path in paths {
                match read_pdf_from_path(path) {
                    Ok(entry) => inputs.push(entry),
                    Err(err) => errors.push(err),
                }
            }

            let done_counter = AtomicUsize::new(0);
            let total_inputs = inputs.len().max(1);
            let (items, errors) = Self::inspect_loaded_inputs(inputs, errors, &|| {
                let done = done_counter.fetch_add(1, Ordering::Relaxed) + 1;
                let _ = tx.send(WorkerEvent::Progress(ProgressUpdate {
                    current: done.min(total_inputs),
                    total: total_inputs,
                }));
                repaint_ctx.request_repaint();
            });

            let _ = tx.send(WorkerEvent::LoadFinished { items, errors });
            repaint_ctx.request_repaint();
        });
    }

    #[cfg(test)]
    fn load_pdf_inputs(
        &mut self,
        inputs: Vec<(Option<PathBuf>, String, Vec<u8>)>,
        errors: Vec<String>,
    ) {
        let (loaded_items, errors) = Self::inspect_loaded_inputs(inputs, errors, &|| {});
        self.apply_loaded_items(loaded_items, errors);
    }

    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        if self.is_busy() {
            return;
        }

        let dropped = ctx.input(|i| i.raw.dropped_files.clone());
        if dropped.is_empty() {
            return;
        }

        let mut paths = Vec::new();
        let mut inputs = Vec::new();
        let mut errors = Vec::new();

        for file in dropped {
            let path = file.path().to_path_buf();
            if !path.as_os_str().is_empty() && path.exists() {
                let is_pdf = path
                    .extension()
                    .and_then(|e| e.to_str())
                    .is_some_and(|e| e.eq_ignore_ascii_case("pdf"));
                if is_pdf {
                    paths.push(path);
                } else {
                    errors.push(format!(
                        "Unsupported file dropped ({}). Please select a .pdf file.",
                        path.display()
                    ));
                }
            } else if let Ok(bytes) = file.bytes() {
                let name = path
                    .file_name()
                    .and_then(|n| n.to_str())
                    .map(|s| s.to_string())
                    .unwrap_or_else(|| "document.pdf".to_string());
                inputs.push((None, name, bytes));
            }
        }

        if paths.is_empty() && inputs.is_empty() {
            if !errors.is_empty() {
                self.error_message = Some(errors.join("; "));
            }
            return;
        }

        self.start_loading_job(paths, inputs, errors, ctx);
    }

    fn handle_row_click(&mut self, index: usize, modifiers: egui::Modifiers) {
        if index >= self.files.len() {
            return;
        }

        if modifiers.shift {
            let anchor = self
                .selection_anchor
                .unwrap_or(0)
                .min(self.files.len().saturating_sub(1));
            let (start, end) = if anchor <= index {
                (anchor, index)
            } else {
                (index, anchor)
            };
            self.selected_indices = (start..=end).collect();
            if self.selection_anchor.is_none() {
                self.selection_anchor = Some(anchor);
            }
        } else if modifiers.command {
            if !self.selected_indices.remove(&index) {
                self.selected_indices.insert(index);
            }
            self.selection_anchor = Some(index);
        } else {
            self.selected_indices.clear();
            self.selected_indices.insert(index);
            self.selection_anchor = Some(index);
        }
    }

    fn select_all(&mut self) {
        if self.files.is_empty() {
            return;
        }
        self.selected_indices = (0..self.files.len()).collect();
        if self.selection_anchor.is_none_or(|a| a >= self.files.len()) {
            self.selection_anchor = Some(0);
        }
    }

    fn sync_all_from_active(&mut self) {
        let Some(active_idx) = self.active_index() else {
            return;
        };
        let template = self.files[active_idx].settings.clone();
        self.shared_target_size_mb = Some(template.target_size_mb.max(0.0));
        for (idx, item) in self.files.iter_mut().enumerate() {
            if idx == active_idx {
                recalculate_item_with_delta(item, None);
                continue;
            }
            item.settings = template.clamped_for_file(item.original_size_mb as f32);
            recalculate_item_with_delta(item, None);
        }
    }

    fn affected_indices(&self) -> Vec<usize> {
        if self.files.is_empty() {
            return Vec::new();
        }
        if self.is_multi_file() && self.apply_for_all {
            (0..self.files.len()).collect()
        } else {
            self.selected_indices
                .iter()
                .copied()
                .filter(|&i| i < self.files.len())
                .collect()
        }
    }

    fn apply_settings_delta(&mut self, draft: &CompressionSettings, delta: SettingsDelta) {
        if !delta.any_changed() {
            return;
        }

        if self.files.is_empty() {
            self.unloaded_settings = draft.clamped_for_file(10.0);
            return;
        }

        let apply_all = self.is_multi_file() && self.apply_for_all;
        if apply_all && delta.target_mb_changed {
            self.shared_target_size_mb = Some(draft.target_size_mb.max(0.0));
        } else if apply_all
            && delta.target_mode_changed
            && draft.target_mode == TargetMode::Megabytes
            && self.shared_target_size_mb.is_none()
        {
            self.shared_target_size_mb = Some(draft.target_size_mb.max(0.0));
        }

        let shared_mb = self.shared_target_size_mb;
        let target_indices = self.affected_indices();
        for idx in target_indices {
            let item = &mut self.files[idx];
            let max_mb = item.original_size_mb as f32;

            if delta.format_changed {
                item.settings.options.format = draft.options.format;
            }
            if delta.quality_changed {
                item.settings.options.quality = draft.options.quality;
            }
            if delta.scale_changed {
                item.settings.options.scale = draft.options.scale;
            }
            if delta.filter_changed {
                item.settings.options.resize_filter = draft.options.resize_filter;
            }
            if delta.target_mode_changed {
                item.settings.target_mode = draft.target_mode;
                match draft.target_mode {
                    TargetMode::Megabytes => {
                        let mb = if apply_all {
                            shared_mb.unwrap_or(item.settings.target_size_mb)
                        } else {
                            item.settings.target_size_mb
                        };
                        item.settings.target_size_mb = mb.clamp(0.0, max_mb);
                    }
                    TargetMode::Percent => {
                        item.settings.target_size_pct = if apply_all {
                            draft.target_size_pct.clamp(1.0, 100.0)
                        } else {
                            initial_target_pct_from_mb(
                                item.settings.target_size_mb,
                                item.original_size_mb,
                            )
                        };
                    }
                    TargetMode::Off => {}
                }
            }
            if delta.target_mb_changed {
                item.settings.target_size_mb = draft.target_size_mb.clamp(0.0, max_mb);
            }
            if delta.target_pct_changed {
                item.settings.target_size_pct = draft.target_size_pct.clamp(1.0, 100.0);
            }

            recalculate_item_with_delta(item, Some(delta));
        }
    }

    fn start_compression(&mut self, ctx: &egui::Context) {
        if self.files.is_empty() || self.is_busy() {
            return;
        }

        if self.files.len() == 1 {
            self.start_single_compression(ctx);
        } else {
            self.start_batch_compression(ctx);
        }
    }

    fn start_single_compression(&mut self, ctx: &egui::Context) {
        let Some(item) = self.files.first_mut() else {
            return;
        };

        self.error_message = None;
        self.status_message = None;
        self.batch_saved_folder = None;
        item.result = None;
        item.error = None;

        let bytes = Arc::clone(&item.bytes);
        let options = item.settings.options.clone();
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress = Some(ProgressUpdate {
            current: 1,
            total: 1,
        });

        let repaint_ctx = ctx.clone();
        thread::spawn(move || {
            let result = compress_pdf(&bytes, &options, |_| {
                repaint_ctx.request_repaint();
            })
            .map_err(|e| e.to_string());

            let _ = tx.send(WorkerEvent::SingleFinished(result));
            repaint_ctx.request_repaint();
        });
    }

    fn start_batch_compression(&mut self, ctx: &egui::Context) {
        let mut dialog = FileDialog::new();
        if let Some(dir) = self
            .active_index()
            .and_then(|idx| self.files.get(idx))
            .or_else(|| self.files.first())
            .and_then(|f| f.path.as_ref())
            .and_then(|p| p.parent())
        {
            dialog = dialog.set_directory(dir);
        }

        let Some(folder) = dialog.pick_folder() else {
            self.status_message = Some("Batch save cancelled.".to_string());
            return;
        };

        self.error_message = None;
        self.status_message = None;
        self.batch_saved_folder = None;
        for item in &mut self.files {
            item.result = None;
            item.error = None;
        }

        let output_paths = unique_batch_output_paths(&self.files, &folder);
        let jobs: Vec<BatchJob> = self
            .files
            .iter()
            .zip(output_paths)
            .enumerate()
            .map(|(index, (item, output_path))| BatchJob {
                index,
                original_size_mb: item.original_size_mb,
                bytes: Arc::clone(&item.bytes),
                settings: item.settings.clone(),
                output_path,
            })
            .collect();

        let total_jobs = jobs.len().max(1);
        let (tx, rx) = mpsc::channel();
        self.worker_rx = Some(rx);
        self.progress = Some(ProgressUpdate {
            current: 1,
            total: total_jobs,
        });

        let repaint_ctx = ctx.clone();
        thread::spawn(move || {
            jobs.into_iter().for_each(|job| {
                let current_idx = (job.index + 1).min(total_jobs);
                let _ = tx.send(WorkerEvent::Progress(ProgressUpdate {
                    current: current_idx,
                    total: total_jobs,
                }));
                repaint_ctx.request_repaint();

                let outcome = compress_pdf(&job.bytes, &job.settings.options, |_| {})
                    .map_err(|e| e.to_string())
                    .and_then(|(out_bytes, stats)| {
                        std::fs::write(&job.output_path, &out_bytes).map_err(|e| {
                            format!("Failed to save {}: {e}", job.output_path.display())
                        })?;
                        Ok(CompressionResult::from_stats(
                            job.output_path,
                            stats,
                            &job.settings,
                            job.original_size_mb,
                        ))
                    });

                let _ = tx.send(WorkerEvent::BatchItemFinished {
                    index: job.index,
                    outcome,
                });
                repaint_ctx.request_repaint();
            });

            let _ = tx.send(WorkerEvent::BatchFinished { folder });
            repaint_ctx.request_repaint();
        });
    }

    fn poll_worker(&mut self) {
        let Some(rx) = &self.worker_rx else {
            return;
        };

        let mut events = Vec::new();
        while let Ok(event) = rx.try_recv() {
            events.push(event);
        }

        for event in events {
            match event {
                WorkerEvent::Progress(p) => {
                    self.progress = Some(p);
                }
                WorkerEvent::LoadFinished { items, errors } => {
                    self.worker_rx = None;
                    self.progress = None;
                    self.apply_loaded_items(items, errors);
                }
                WorkerEvent::SingleFinished(result) => {
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
                WorkerEvent::BatchItemFinished { index, outcome } => {
                    if let Some(item) = self.files.get_mut(index) {
                        match outcome {
                            Ok(res) => {
                                item.result = Some(res);
                                item.error = None;
                            }
                            Err(err) => {
                                item.result = None;
                                item.error = Some(err);
                            }
                        }
                    }
                }
                WorkerEvent::BatchFinished { folder } => {
                    self.worker_rx = None;
                    self.progress = None;
                    self.batch_saved_folder = Some(folder);
                    let failed: Vec<String> = self
                        .files
                        .iter()
                        .filter_map(|f| f.error.as_ref().map(|e| format!("{}: {e}", f.file_name)))
                        .collect();
                    if !failed.is_empty() {
                        self.error_message = Some(format!(
                            "{} file(s) failed during batch export: {}",
                            failed.len(),
                            failed.join("; ")
                        ));
                    } else {
                        let total_imgs: usize = self
                            .files
                            .iter()
                            .filter_map(|f| f.result.as_ref())
                            .map(|r| r.stats.total_image_objects)
                            .sum();
                        let skipped_unsupported: usize = self
                            .files
                            .iter()
                            .filter_map(|f| f.result.as_ref())
                            .map(|r| r.stats.images_skipped_unsupported)
                            .sum();
                        if skipped_unsupported > 0 {
                            self.status_message = Some(format!(
                                "Batch complete ({total_imgs} images scanned; {skipped_unsupported} unsupported skipped)."
                            ));
                        }
                    }
                }
            }
        }
    }

    fn build_pending_save(&self, bytes: Vec<u8>, stats: CompressionStats) -> PendingSave {
        let (default_file_name, initial_dir) = match self.files.first() {
            Some(item) => {
                let stem = item_output_stem(item);
                let dir = item
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

        let summary_note = format_compression_summary_note(&pending.stats);
        match dialog.save_file() {
            Some(save_path) => match std::fs::write(&save_path, &pending.bytes) {
                Ok(()) => {
                    self.status_message = Some(format!(
                        "Saved compressed PDF to {} ({summary_note})",
                        save_path.display()
                    ));
                    if let Some(item) = self.files.first_mut() {
                        item.result = Some(CompressionResult::from_stats(
                            save_path,
                            pending.stats,
                            &item.settings,
                            item.original_size_mb,
                        ));
                    }
                }
                Err(err) => {
                    self.error_message =
                        Some(format!("Compression succeeded, but saving failed: {err}"));
                }
            },
            None => {
                self.status_message = Some(format!(
                    "Save cancelled (compressed output would have been {}; {summary_note}).",
                    format_bytes(pending.stats.compressed_bytes)
                ));
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
                        let ctx = ui.ctx().clone();
                        self.open_file_dialog(&ctx);
                    }

                    ui.add_space(10.0);

                    const COMPRESS_BTN_WIDTH: f32 = 152.0;
                    let middle_w = (ui.available_width() - COMPRESS_BTN_WIDTH - 12.0).max(80.0);

                    if self.is_multi_file() {
                        ui.allocate_ui_with_layout(
                            egui::vec2(middle_w, 38.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                let interact_h = ui.spacing().interact_size.y;
                                ui.add_space(((38.0 - interact_h) * 0.5).max(0.0));
                                let prev = self.apply_for_all;
                                let resp = ui.add_enabled(
                                    !busy,
                                    egui::Checkbox::new(&mut self.apply_for_all, "Apply for ALL"),
                                );
                                if resp.changed() && self.apply_for_all && !prev {
                                    self.sync_all_from_active();
                                } else if resp.changed() && !self.apply_for_all {
                                    self.shared_target_size_mb = None;
                                }
                            },
                        );
                    } else {
                        let single = self.files.first();
                        let middle_layout = if single.is_some() {
                            egui::Layout::top_down(egui::Align::Min)
                        } else {
                            egui::Layout::left_to_right(egui::Align::Center)
                        };
                        ui.allocate_ui_with_layout(
                            egui::vec2(middle_w, 38.0),
                            middle_layout,
                            |ui| {
                                if let Some(loaded) = single {
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
                                                loaded.summary.page_count,
                                                loaded.summary.image_count
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
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_compress = !self.files.is_empty() && !busy;
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

                if self.is_multi_file() {
                    ui.add_space(10.0);
                    self.render_batch_file_table(ui, busy);
                }
            });
    }

    fn render_batch_file_table(&mut self, ui: &mut egui::Ui, busy: bool) {
        let dark = ui.visuals().dark_mode;
        let total_w = ui.available_width();
        let widths = file_table_col_widths(total_w);
        let aligns = [
            egui::Align::Center,
            egui::Align::Min,
            egui::Align::Max,
            egui::Align::Max,
        ];
        let est_header = if self.est_show_percent {
            "Est. (%)"
        } else {
            "Est. (MB)"
        };

        if render_table_header_row_interactive(
            ui,
            total_w,
            widths,
            aligns,
            ["No.", "Name", "Size", est_header],
            Some((
                3,
                ui.scope_id().with("batch_file_est_col_header"),
                "Click to toggle MB / %",
            )),
        ) {
            self.est_show_percent = !self.est_show_percent;
        }
        ui.add_space(3.0);

        let max_scroll_h = (FILE_TABLE_MAX_VISIBLE_ROWS as f32) * TABLE_ROW_HEIGHT
            + ((FILE_TABLE_MAX_VISIBLE_ROWS - 1) as f32) * TABLE_ROW_GAP;

        let mut clicked_row: Option<(usize, egui::Modifiers)> = None;
        let guide_stroke = Stroke::new(1.0, table_row_guide_color(dark));

        egui::ScrollArea::vertical()
            .id_salt("batch_file_list_scroll")
            .max_height(max_scroll_h)
            .auto_shrink([false, true])
            .show(ui, |ui| {
                ui.spacing_mut().item_spacing.y = TABLE_ROW_GAP;
                let row_w = total_w.min(ui.available_width());
                let row_widths = file_table_col_widths(row_w);

                for (i, item) in self.files.iter().enumerate() {
                    let is_selected = self.selected_indices.contains(&i);
                    let (row_rect, row_resp) = ui.allocate_exact_size(
                        egui::vec2(row_w, TABLE_ROW_HEIGHT),
                        if busy {
                            egui::Sense::hover()
                        } else {
                            egui::Sense::click()
                        },
                    );

                    let bg = if is_selected {
                        if dark {
                            Color32::from_rgb(42, 74, 118)
                        } else {
                            Color32::from_rgb(214, 232, 255)
                        }
                    } else if row_resp.hovered() && !busy {
                        if dark {
                            Color32::from_rgb(38, 43, 52)
                        } else {
                            Color32::from_rgb(235, 240, 248)
                        }
                    } else if i % 2 == 1 {
                        if dark {
                            Color32::from_rgba_unmultiplied(255, 255, 255, 5)
                        } else {
                            Color32::from_rgba_unmultiplied(0, 0, 0, 5)
                        }
                    } else {
                        Color32::TRANSPARENT
                    };

                    if bg != Color32::TRANSPARENT {
                        ui.painter()
                            .rect_filled(row_rect, CornerRadius::same(4), bg);
                    }
                    if is_selected {
                        ui.painter().rect_stroke(
                            row_rect,
                            CornerRadius::same(4),
                            Stroke::new(1.0, Color32::from_rgb(66, 133, 244)),
                            egui::StrokeKind::Inside,
                        );
                    }

                    let no_text = format!("{}", i + 1);
                    let size_text = format_mb_size(item.summary.file_bytes);
                    let est_bytes = item.estimated_compressed_bytes();
                    let est_text = format_file_est_cell(
                        item.summary.file_bytes,
                        est_bytes,
                        self.est_show_percent,
                    );

                    let text_color = if is_selected {
                        Some(ui.visuals().strong_text_color())
                    } else {
                        None
                    };

                    paint_row_cells(
                        ui,
                        row_rect,
                        &TableRowCells {
                            widths: row_widths,
                            aligns,
                            texts: [&no_text, &item.file_name, &size_text, &est_text],
                            colors: [text_color, text_color, text_color, text_color],
                            strong: [false, is_selected, false, false],
                            font_size: 12.5,
                            divider_stroke: guide_stroke,
                            divider_v_inset: 3.0,
                        },
                    );

                    if row_resp.clicked() && !busy {
                        let modifiers = ui.ctx().input(|inp| inp.modifiers);
                        clicked_row = Some((i, modifiers));
                    }

                    row_resp.on_hover_text(&item.file_name);
                }
            });

        if let Some((index, modifiers)) = clicked_row {
            self.handle_row_click(index, modifiers);
        }
    }

    fn render_controls_card(&mut self, ui: &mut egui::Ui, busy: bool) {
        let dark = ui.visuals().dark_mode;
        let card_fill = neutral_card_fill(dark);
        let border_color = neutral_card_border(dark);

        let active_idx = self.active_index();
        let has_files = !self.files.is_empty();
        let has_editable_target =
            !has_files || (self.is_multi_file() && self.apply_for_all) || !self.selected_indices.is_empty();

        let mut draft = match active_idx {
            Some(idx) => {
                let mut s = self.files[idx].settings.clone();
                if self.is_multi_file() && self.apply_for_all {
                    if let Some(shared) = self.shared_target_size_mb {
                        s.target_size_mb = shared;
                    }
                }
                s
            }
            None => self.unloaded_settings.clone(),
        };

        let max_target_mb = if !has_files {
            10.0
        } else if self.is_multi_file() && self.apply_for_all {
            self.files
                .iter()
                .map(|f| f.original_size_mb as f32)
                .fold(0.0_f32, f32::max)
                .max(1e-6)
        } else {
            active_idx
                .map(|idx| (self.files[idx].original_size_mb as f32).max(1e-6))
                .unwrap_or(10.0)
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

                let mut delta = SettingsDelta::default();

                ui.add_enabled_ui(!busy && has_editable_target, |ui| {
                    // Format / Algorithm dropdown
                    let prev_format = draft.options.format;
                    ui.horizontal(|ui| {
                        ui.label("Image Format:");
                        egui::ComboBox::from_id_salt("image_format_selector")
                            .selected_text(draft.options.format.label())
                            .width(105.0)
                            .show_ui(ui, |ui| {
                                for fmt in ImageFormatChoice::ALL {
                                    ui.selectable_value(
                                        &mut draft.options.format,
                                        fmt,
                                        fmt.label(),
                                    );
                                }
                            });
                    });
                    if draft.options.format != prev_format {
                        delta.format_changed = true;
                    }

                    ui.add_space(10.0);

                    // Quality Factor + Resolution Scale aligned cleanly in a 2-column grid
                    let uses_quality = draft.options.format.uses_quality();
                    let prev_quality = draft.options.quality;
                    let prev_scale = draft.options.scale;
                    let item_gap = ui.spacing().item_spacing.x;
                    let value_box_w = 56.0_f32;
                    let drag_h = ui.spacing().interact_size.y;
                    let slider_track_w =
                        (ui.available_width() - 122.0 - 8.0 - value_box_w - item_gap)
                            .clamp(120.0, 260.0);
                    #[cfg(test)]
                    let mut captured_q_slider = None;
                    #[cfg(test)]
                    let mut captured_q_drag = None;
                    #[cfg(test)]
                    let mut captured_s_slider = None;
                    #[cfg(test)]
                    let mut captured_s_drag = None;

                    egui::Grid::new("quality_scale_sliders_grid")
                        .num_columns(2)
                        .spacing([8.0, 8.0])
                        .show(ui, |ui| {
                            ui.label("Quality Factor:");
                            ui.horizontal(|ui| {
                                ui.spacing_mut().slider_width = slider_track_w;
                                let slider = egui::Slider::new(&mut draft.options.quality, 1..=100)
                                    .show_value(false)
                                    .clamping(egui::SliderClamping::Always);
                                let quality_resp = ui
                                    .add_enabled(uses_quality, slider)
                                    .on_disabled_hover_text(
                                        "Lossless mode — Quality Factor is ignored",
                                    );
                                let quality_drag_resp = ui
                                    .add_enabled_ui(uses_quality, |ui| {
                                        ui.add_sized(
                                            [value_box_w, drag_h],
                                            egui::DragValue::new(&mut draft.options.quality)
                                                .range(1..=100)
                                                .speed(0.5),
                                        )
                                    })
                                    .inner
                                    .on_disabled_hover_text(
                                        "Lossless mode — Quality Factor is ignored",
                                    );
                                #[cfg(test)]
                                {
                                    captured_q_slider = Some(quality_resp.rect);
                                    captured_q_drag = Some(quality_drag_resp.rect);
                                }
                                if uses_quality
                                    && (quality_resp.changed() || quality_drag_resp.changed())
                                    && draft.options.quality != prev_quality
                                {
                                    delta.quality_changed = true;
                                }
                            });
                            ui.end_row();

                            ui.label("Resolution Scale:");
                            ui.horizontal(|ui| {
                                ui.spacing_mut().slider_width = slider_track_w;
                                let scale_slider_resp = ui.add(
                                    egui::Slider::new(&mut draft.options.scale, 0.10..=1.00)
                                        .step_by(0.01)
                                        .show_value(false)
                                        .clamping(egui::SliderClamping::Edits),
                                );
                                let scale_drag_resp = ui.add_sized(
                                    [value_box_w, drag_h],
                                    egui::DragValue::new(&mut draft.options.scale)
                                        .range(0.10..=1.00)
                                        .speed(0.01)
                                        .fixed_decimals(2)
                                        .suffix("x"),
                                );
                                #[cfg(test)]
                                {
                                    captured_s_slider = Some(scale_slider_resp.rect);
                                    captured_s_drag = Some(scale_drag_resp.rect);
                                }
                                if (scale_slider_resp.changed() || scale_drag_resp.changed())
                                    && (draft.options.scale - prev_scale).abs() >= 1e-4
                                {
                                    delta.scale_changed = true;
                                }
                            });
                            ui.end_row();
                        });
                    #[cfg(test)]
                    {
                        self.last_quality_slider_rect = captured_q_slider;
                        self.last_quality_drag_rect = captured_q_drag;
                        self.last_scale_slider_rect = captured_s_slider;
                        self.last_scale_drag_rect = captured_s_drag;
                    }

                    ui.add_space(8.0);

                    // Resize filter choice
                    let prev_filter = draft.options.resize_filter;
                    ui.horizontal(|ui| {
                        ui.label("Resize Filter:");
                        for filter in ResizeFilterChoice::ALL {
                            let selected = draft.options.resize_filter == filter;
                            if ui.selectable_label(selected, filter.label()).clicked() {
                                draft.options.resize_filter = filter;
                            }
                        }
                    });
                    if draft.options.resize_filter != prev_filter {
                        delta.filter_changed = true;
                    }

                    ui.add_space(8.0);

                    // Unified 3-state Target Size (Off -> MB -> %) / Estimated Size row
                    ui.horizontal(|ui| {
                        let prev_mode = draft.target_mode;
                        let mut cb_checked = prev_mode.is_enabled();
                        let cb_resp = ui.checkbox(&mut cb_checked, prev_mode.label());
                        if cb_resp.changed() {
                            let next_mode = prev_mode.next();
                            draft.target_mode = next_mode;
                            delta.target_mode_changed = true;
                            delta.target_mode_activated = next_mode.is_enabled();
                            if prev_mode == TargetMode::Megabytes
                                && next_mode == TargetMode::Percent
                            {
                                draft.target_size_pct = active_idx
                                    .map(|idx| {
                                        initial_target_pct_from_mb(
                                            draft.target_size_mb,
                                            self.files[idx].original_size_mb,
                                        )
                                    })
                                    .unwrap_or(100.0);
                            }
                        }

                        match draft.target_mode {
                            TargetMode::Off => {
                                ui.add_space(6.0);
                                self.render_estimated_size_inline(ui, active_idx, &draft.options);
                            }
                            TargetMode::Megabytes => {
                                let target_decimals = if has_files {
                                    target_decimals_for_mb(f64::from(max_target_mb))
                                } else {
                                    2
                                };
                                draft.target_size_mb =
                                    draft.target_size_mb.clamp(0.0, max_target_mb);
                                let prev_target = draft.target_size_mb;

                                ui.add_enabled_ui(has_files, |ui| {
                                    let slider_resp = ui.add(
                                        egui::Slider::new(
                                            &mut draft.target_size_mb,
                                            0.0..=max_target_mb,
                                        )
                                        .show_value(false)
                                        .clamping(egui::SliderClamping::Always),
                                    );
                                    let drag_resp = ui.add(
                                        egui::DragValue::new(&mut draft.target_size_mb)
                                            .range(0.0..=max_target_mb)
                                            .speed(
                                                (f64::from(max_target_mb) / 200.0)
                                                    .clamp(0.001, 0.25),
                                            )
                                            .fixed_decimals(target_decimals)
                                            .suffix(" MB"),
                                    );
                                    if (slider_resp.changed() || drag_resp.changed())
                                        && (draft.target_size_mb - prev_target).abs() >= 1e-5
                                    {
                                        delta.target_mb_changed = true;
                                    }
                                });

                                if has_files && draft.target_size_mb <= 0.0 {
                                    ui.label(
                                        RichText::new("(No limit)")
                                            .size(12.0)
                                            .color(ui.visuals().weak_text_color()),
                                    );
                                }
                            }
                            TargetMode::Percent => {
                                draft.target_size_pct = draft.target_size_pct.clamp(1.0, 100.0);
                                let prev_pct = draft.target_size_pct;

                                ui.add_enabled_ui(has_files, |ui| {
                                    let slider_resp = ui.add(
                                        egui::Slider::new(
                                            &mut draft.target_size_pct,
                                            1.0..=100.0,
                                        )
                                        .show_value(false)
                                        .step_by(1.0)
                                        .clamping(egui::SliderClamping::Always),
                                    );
                                    let drag_resp = ui.add(
                                        egui::DragValue::new(&mut draft.target_size_pct)
                                            .range(1.0..=100.0)
                                            .speed(0.5)
                                            .fixed_decimals(0)
                                            .suffix("%"),
                                    );
                                    if (slider_resp.changed() || drag_resp.changed())
                                        && (draft.target_size_pct - prev_pct).abs() >= 1e-4
                                    {
                                        delta.target_pct_changed = true;
                                    }
                                });
                            }
                        }
                    });
                });

                if delta.any_changed() {
                    self.apply_settings_delta(&draft, delta);
                    ui.ctx().request_repaint();
                }
            });
    }

    fn render_estimated_size_inline(
        &self,
        ui: &mut egui::Ui,
        active_idx: Option<usize>,
        _options: &CompressionOptions,
    ) {
        let Some(item) = active_idx.and_then(|idx| self.files.get(idx)) else {
            ui.label(
                RichText::new("Estimated Size: —")
                    .size(12.5)
                    .color(ui.visuals().weak_text_color()),
            );
            return;
        };

        ui.label(
            RichText::new(estimated_size_text(item))
                .strong()
                .color(ui.visuals().strong_text_color()),
        );
    }

    fn single_summary_metrics(&self) -> (String, String, String, bool) {
        let Some(item) = self.files.first() else {
            return ("-".to_string(), "-".to_string(), "-".to_string(), false);
        };

        match &item.result {
            Some(result) => (
                format_bytes(result.stats.original_bytes),
                format_bytes(result.stats.compressed_bytes),
                format_saved_pct(result.space_saved_pct),
                false,
            ),
            None => {
                let orig_bytes = item.summary.file_bytes;
                let (saved_str, is_est) = match item.estimated_saved_percentage() {
                    Some(saved_pct) => (format_estimated_saved_pct(saved_pct), true),
                    None => ("-".to_string(), false),
                };
                (format_bytes(orig_bytes), "-".to_string(), saved_str, is_est)
            }
        }
    }

    fn render_summary_card(&self, ui: &mut egui::Ui) {
        if self.is_multi_file() {
            self.render_batch_summary_card(ui);
        } else {
            self.render_single_summary_card(ui);
        }
    }

    fn render_single_summary_card(&self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        let item = self.files.first();
        let result = item.and_then(|f| f.result.as_ref());
        let has_active_target = item
            .and_then(|f| f.settings.effective_target_mb(f.original_size_mb))
            .is_some_and(|mb| mb_to_bytes(mb) > 0);
        let over_target = result.is_some_and(|r| r.over_target);

        let (card_fill, border_color) = match result {
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
                    if self.is_busy() {
                        self.render_summary_header_progress(ui);
                    } else if result.is_some() {
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

                let (orig_text, new_text, saved_text, saved_is_estimate) =
                    self.single_summary_metrics();

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

                if let Some(res) = result {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("Saved to: {}", res.saved_path.display()))
                            .size(11.5)
                            .color(ui.visuals().weak_text_color()),
                    );
                }
            });
    }

    fn render_summary_header_progress(&self, ui: &mut egui::Ui) {
        let Some(p) = &self.progress else {
            return;
        };
        let total = p.total.max(1);
        let current = p.current.min(total);
        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
            ui.spacing_mut().item_spacing.x = 6.0;
            ui.label(
                RichText::new(format!("{current}/{total}"))
                    .size(12.5)
                    .strong()
                    .color(ui.visuals().strong_text_color()),
            );
            ui.add(egui::Spinner::new().size(14.0));
        });
    }

    fn render_batch_summary_card(&self, ui: &mut egui::Ui) {
        let dark = ui.visuals().dark_mode;
        Frame::new()
            .fill(neutral_card_fill(dark))
            .stroke(Stroke::new(1.0, neutral_card_border(dark)))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Compression Summary").strong().size(15.0));
                    if self.is_busy() {
                        self.render_summary_header_progress(ui);
                    }
                });
                ui.add_space(8.0);

                let total_w = ui.available_width();
                let header_widths = summary_table_col_widths(total_w);
                let aligns = [
                    egui::Align::Center,
                    egui::Align::Min,
                    egui::Align::Min,
                    egui::Align::Min,
                ];

                render_table_header_row(
                    ui,
                    total_w,
                    header_widths,
                    aligns,
                    ["No.", "Original Size", "New Size", "Space Saved"],
                );
                ui.add_space(3.0);

                let max_scroll_h = (SUMMARY_TABLE_MAX_VISIBLE_ROWS as f32) * TABLE_ROW_HEIGHT
                    + ((SUMMARY_TABLE_MAX_VISIBLE_ROWS - 1) as f32) * TABLE_ROW_GAP;
                let guide_stroke = Stroke::new(1.0, table_row_guide_color(dark));

                egui::ScrollArea::vertical()
                    .id_salt("batch_summary_scroll")
                    .max_height(max_scroll_h)
                    .auto_shrink([false, true])
                    .show(ui, |ui| {
                        ui.spacing_mut().item_spacing.y = TABLE_ROW_GAP;
                        let row_w = total_w.min(ui.available_width());
                        let widths = summary_table_col_widths(row_w);

                        for (i, item) in self.files.iter().enumerate() {
                            let (row_bg, row_text_color) = batch_row_colors(dark, item);
                            let (row_rect, _) = ui.allocate_exact_size(
                                egui::vec2(row_w, TABLE_ROW_HEIGHT),
                                egui::Sense::hover(),
                            );

                            if row_bg != Color32::TRANSPARENT {
                                ui.painter()
                                    .rect_filled(row_rect, CornerRadius::same(4), row_bg);
                            }

                            let no_text = format!("{}", i + 1);
                            let orig_text = format_mb_size(item.summary.file_bytes);

                            let (new_text, saved_text, saved_color, saved_bold) =
                                match (&item.result, &item.error) {
                                    (Some(res), _) => (
                                        format!("{:.2} MB", res.new_size_mb),
                                        format_saved_pct(res.space_saved_pct),
                                        row_text_color,
                                        true,
                                    ),
                                    (None, Some(_err)) => {
                                        ("Failed".to_string(), "-".to_string(), row_text_color, true)
                                    }
                                    (None, None) => {
                                        let saved_s = match item.estimated_saved_percentage() {
                                            Some(pct) => format_estimated_saved_pct(pct),
                                            None => "-".to_string(),
                                        };
                                        (
                                            "-".to_string(),
                                            saved_s,
                                            Some(ui.visuals().weak_text_color()),
                                            false,
                                        )
                                    }
                                };

                            paint_row_cells(
                                ui,
                                row_rect,
                                &TableRowCells {
                                    widths,
                                    aligns,
                                    texts: [&no_text, &orig_text, &new_text, &saved_text],
                                    colors: [
                                        row_text_color,
                                        row_text_color,
                                        row_text_color,
                                        saved_color,
                                    ],
                                    strong: [false, false, item.result.is_some(), saved_bold],
                                    font_size: 12.5,
                                    divider_stroke: guide_stroke,
                                    divider_v_inset: 3.0,
                                },
                            );
                        }
                    });

                if let Some(folder) = &self.batch_saved_folder {
                    ui.add_space(8.0);
                    ui.label(
                        RichText::new(format!("Saved to: {}", folder.display()))
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
        if busy {
            ctx.request_repaint();
        }
        if self.is_multi_file() && !busy && !ctx.egui_wants_keyboard_input() {
            let select_all = ctx.input_mut(|i| {
                i.focused
                    && i.consume_shortcut(&egui::KeyboardShortcut::new(
                        egui::Modifiers::COMMAND,
                        egui::Key::A,
                    ))
            });
            if select_all {
                self.select_all();
            }
        }

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

                    let single_completed = self.files.len() == 1
                        && self.files.first().is_some_and(|f| f.result.is_some());
                    if !single_completed && self.batch_saved_folder.is_none() {
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

        #[cfg(test)]
        {
            self.last_content_width = panel_resp.inner.content_size.x;
        }
        let content_height = panel_resp.inner.content_size.y;
        if content_height.is_finite() && content_height > 0.0 {
            let max_fit_h = ctx
                .input(|i| {
                    i.viewport()
                        .monitor_size
                        .map(|m| (m.y - 80.0).max(MIN_WINDOW_HEIGHT))
                })
                .unwrap_or(960.0);
            let ideal_height = (content_height + panel_v_margin + 2.0)
                .ceil()
                .min(max_fit_h);
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

fn initial_target_pct_from_mb(target_size_mb: f32, original_size_mb: f64) -> f32 {
    if target_size_mb > 0.0 && original_size_mb > 0.0 {
        ((f64::from(target_size_mb) / original_size_mb) * 100.0)
            .round()
            .clamp(1.0, 100.0) as f32
    } else {
        100.0
    }
}

fn recalculate_item_with_delta(item: &mut PdfFileItem, delta: Option<SettingsDelta>) {
    if let Some(effective_mb) = item.settings.effective_target_mb(item.original_size_mb) {
        let target_bytes = mb_to_bytes(effective_mb);
        if target_bytes > 0 {
            let at_default_full_bounds = target_bytes >= item.summary.file_bytes
                && item.settings.options.quality == 100
                && (item.settings.options.scale - 1.0).abs() < 1e-4
                && delta.is_some_and(|d| !(d.format_changed || d.filter_changed));

            if !at_default_full_bounds {
                match delta {
                    None => {
                        item.summary
                            .apply_target_scale_first(&mut item.settings.options, target_bytes);
                    }
                    Some(d)
                        if d.target_mode_activated
                            || d.target_mb_changed
                            || d.target_pct_changed
                            || d.format_changed
                            || d.filter_changed =>
                    {
                        item.summary
                            .apply_target_scale_first(&mut item.settings.options, target_bytes);
                    }
                    Some(d) if d.quality_changed => {
                        item.settings.options.scale = item
                            .summary
                            .solve_scale_for_target(&item.settings.options, target_bytes);
                    }
                    Some(d) if d.scale_changed && item.settings.options.format.uses_quality() => {
                        item.settings.options.quality = item
                            .summary
                            .solve_quality_for_target(&item.settings.options, target_bytes);
                    }
                    _ => {}
                }
            }
        }
    }

    item.cached_estimated_bytes = item.summary.estimate_compressed_bytes(&item.settings.options);
}

fn item_output_stem(item: &PdfFileItem) -> String {
    item.path
        .as_deref()
        .and_then(Path::file_stem)
        .or_else(|| Path::new(&item.file_name).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "document".to_string())
}

fn unique_batch_output_paths(files: &[PdfFileItem], folder: &Path) -> Vec<PathBuf> {
    let mut used = HashSet::with_capacity(files.len());
    let mut paths = Vec::with_capacity(files.len());

    for item in files {
        let stem = item_output_stem(item);
        let mut candidate = format!("{stem}_compressed.pdf");
        let mut suffix = 2usize;
        while !used.insert(candidate.to_lowercase()) {
            candidate = format!("{stem}_compressed_{suffix}.pdf");
            suffix += 1;
        }
        paths.push(folder.join(candidate));
    }

    paths
}

fn file_table_col_widths(total_w: f32) -> [f32; 4] {
    let name_w =
        (total_w - FILE_TABLE_NO_COL_WIDTH - FILE_TABLE_SIZE_COL_WIDTH - FILE_TABLE_EST_COL_WIDTH)
            .max(60.0);
    [
        FILE_TABLE_NO_COL_WIDTH,
        name_w,
        FILE_TABLE_SIZE_COL_WIDTH,
        FILE_TABLE_EST_COL_WIDTH,
    ]
}

fn summary_table_col_widths(total_w: f32) -> [f32; 4] {
    let each = ((total_w - TABLE_NO_COL_WIDTH) / 3.0).max(58.0);
    [TABLE_NO_COL_WIDTH, each, each, each]
}

fn table_header_divider_color(dark: bool) -> Color32 {
    if dark {
        Color32::from_gray(65)
    } else {
        Color32::from_gray(205)
    }
}

fn table_row_guide_color(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgba_unmultiplied(255, 255, 255, 16)
    } else {
        Color32::from_rgba_unmultiplied(0, 0, 0, 16)
    }
}

fn table_header_bg(dark: bool) -> Color32 {
    if dark {
        Color32::from_rgba_unmultiplied(255, 255, 255, 6)
    } else {
        Color32::from_rgba_unmultiplied(0, 0, 0, 6)
    }
}

fn split_row_columns<const N: usize>(row_rect: egui::Rect, widths: [f32; N]) -> [egui::Rect; N] {
    let y0 = row_rect.min.y;
    let y1 = row_rect.max.y;
    let mut rects = [egui::Rect::NOTHING; N];
    let mut x = row_rect.min.x;
    for (idx, rect) in rects.iter_mut().enumerate() {
        let next_x = if idx + 1 == N {
            row_rect.max.x
        } else {
            (x + widths[idx]).min(row_rect.max.x)
        };
        *rect = egui::Rect::from_min_max(egui::pos2(x, y0), egui::pos2(next_x, y1));
        x = next_x;
    }
    rects
}

fn render_table_header_row<const N: usize>(
    ui: &mut egui::Ui,
    total_w: f32,
    widths: [f32; N],
    aligns: [egui::Align; N],
    titles: [&str; N],
) {
    let _ = render_table_header_row_interactive(ui, total_w, widths, aligns, titles, None);
}

fn render_table_header_row_interactive<const N: usize>(
    ui: &mut egui::Ui,
    total_w: f32,
    widths: [f32; N],
    aligns: [egui::Align; N],
    titles: [&str; N],
    interactive_col: Option<(usize, egui::Id, &str)>,
) -> bool {
    let dark = ui.visuals().dark_mode;
    let divider_color = table_header_divider_color(dark);
    let (header_rect, _) =
        ui.allocate_exact_size(egui::vec2(total_w, TABLE_HEADER_HEIGHT), egui::Sense::hover());

    let mut clicked = false;
    let mut hovered_col_rect: Option<egui::Rect> = None;
    if let Some((col_idx, interact_id, tooltip)) = interactive_col {
        let col_rects = split_row_columns(header_rect, widths);
        if let Some(&target_rect) = col_rects.get(col_idx) {
            let resp = ui.interact(target_rect, interact_id, egui::Sense::click());
            if resp.hovered() {
                hovered_col_rect = Some(target_rect);
            }
            if resp.clicked() {
                clicked = true;
            }
            resp.on_hover_cursor(egui::CursorIcon::PointingHand)
                .on_hover_text(tooltip);
        }
    }

    let painter = ui.painter();
    painter.rect_filled(header_rect, CornerRadius::same(3), table_header_bg(dark));
    if let Some(col_rect) = hovered_col_rect {
        let hover_fill = if dark {
            Color32::from_rgba_unmultiplied(255, 255, 255, 16)
        } else {
            Color32::from_rgba_unmultiplied(0, 0, 0, 14)
        };
        painter.rect_filled(col_rect, CornerRadius::same(2), hover_fill);
    }
    painter.hline(
        header_rect.x_range(),
        header_rect.max.y,
        Stroke::new(1.0, divider_color),
    );

    paint_row_cells(
        ui,
        header_rect,
        &TableRowCells {
            widths,
            aligns,
            texts: titles,
            colors: [Some(ui.visuals().weak_text_color()); N],
            strong: [true; N],
            font_size: 12.0,
            divider_stroke: Stroke::new(1.0, divider_color),
            divider_v_inset: 4.0,
        },
    );

    clicked
}

struct TableRowCells<'a, const N: usize> {
    widths: [f32; N],
    aligns: [egui::Align; N],
    texts: [&'a str; N],
    colors: [Option<Color32>; N],
    strong: [bool; N],
    font_size: f32,
    divider_stroke: Stroke,
    divider_v_inset: f32,
}

fn paint_row_cells<const N: usize>(
    ui: &egui::Ui,
    row_rect: egui::Rect,
    cells: &TableRowCells<'_, N>,
) {
    let col_rects = split_row_columns(row_rect, cells.widths);
    let painter = ui.painter();
    let div_y0 = row_rect.min.y + cells.divider_v_inset;
    let div_y1 = row_rect.max.y - cells.divider_v_inset;
    for col in &col_rects[..N.saturating_sub(1)] {
        painter.vline(col.max.x, div_y0..=div_y1, cells.divider_stroke);
    }

    let fallback_color = ui.visuals().text_color();
    for col in 0..N {
        let col_rect = col_rects[col];
        let inner_min_x = col_rect.min.x + TABLE_CELL_PAD_X;
        let inner_max_x = (col_rect.max.x - TABLE_CELL_PAD_X).max(inner_min_x + 1.0);
        let cell_rect = egui::Rect::from_min_max(
            egui::pos2(inner_min_x, col_rect.min.y),
            egui::pos2(inner_max_x, col_rect.max.y),
        );

        let mut rt = RichText::new(cells.texts[col]).size(cells.font_size);
        if cells.strong[col] {
            rt = rt.strong();
        }
        if let Some(c) = cells.colors[col] {
            rt = rt.color(c);
        }

        let galley = egui::WidgetText::from(rt).into_galley(
            ui,
            Some(egui::TextWrapMode::Truncate),
            cell_rect.width(),
            egui::TextStyle::Body,
        );

        let x = match cells.aligns[col] {
            egui::Align::Min => cell_rect.min.x,
            egui::Align::Center => cell_rect.center().x - 0.5 * galley.size().x,
            egui::Align::Max => cell_rect.max.x - galley.size().x,
        };
        let y = cell_rect.center().y - 0.5 * galley.size().y;

        painter
            .with_clip_rect(cell_rect)
            .galley(egui::pos2(x, y), galley, fallback_color);
    }
}

fn batch_row_colors(dark: bool, item: &PdfFileItem) -> (Color32, Option<Color32>) {
    match (&item.result, &item.error) {
        (Some(res), _) if res.over_target => {
            if dark {
                (
                    Color32::from_rgb(60, 30, 32),
                    Some(Color32::from_rgb(255, 175, 175)),
                )
            } else {
                (
                    Color32::from_rgb(254, 232, 232),
                    Some(Color32::from_rgb(165, 29, 29)),
                )
            }
        }
        (Some(_), _) => {
            if dark {
                (
                    Color32::from_rgb(26, 52, 37),
                    Some(Color32::from_rgb(166, 244, 192)),
                )
            } else {
                (
                    Color32::from_rgb(228, 248, 234),
                    Some(Color32::from_rgb(21, 110, 52)),
                )
            }
        }
        (None, Some(_)) => {
            if dark {
                (
                    Color32::from_rgb(60, 30, 32),
                    Some(Color32::from_rgb(255, 175, 175)),
                )
            } else {
                (
                    Color32::from_rgb(254, 232, 232),
                    Some(Color32::from_rgb(165, 29, 29)),
                )
            }
        }
        (None, None) => (Color32::TRANSPARENT, None),
    }
}

fn format_estimated_saved_pct(saved_pct: f64) -> String {
    if saved_pct >= 0.0 {
        format!("~{saved_pct:.0}%")
    } else {
        format!("~+{:.0}%", saved_pct.abs())
    }
}

fn estimated_size_text(item: &PdfFileItem) -> String {
    format!("Estimated Size: {}", format_bytes(item.estimated_compressed_bytes()))
}

fn format_mb_size(bytes: usize) -> String {
    format!("{:.2} MB", bytes_to_mb(bytes))
}

fn format_file_est_cell(orig_bytes: usize, est_bytes: usize, show_percent: bool) -> String {
    if show_percent {
        if orig_bytes == 0 {
            "-".to_string()
        } else {
            let pct = (est_bytes as f64 / orig_bytes as f64) * 100.0;
            format!("{pct:.0}%")
        }
    } else {
        format_mb_size(est_bytes)
    }
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

fn read_pdf_from_path(path: PathBuf) -> Result<(Option<PathBuf>, String, Vec<u8>), String> {
    let bytes = std::fs::read(&path).map_err(|err| format!("{} ({err})", path.display()))?;
    let file_name = path
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_else(|| "document.pdf".to_string());
    Ok((Some(path), file_name, bytes))
}

fn format_saved_pct(saved_pct: f64) -> String {
    if saved_pct >= 0.0 {
        format!("{saved_pct:.1}%")
    } else {
        format!("+{:.1}% larger", saved_pct.abs())
    }
}

fn format_compression_summary_note(stats: &CompressionStats) -> String {
    if stats.total_image_objects == 0 {
        return "0 images found; structure optimized".to_string();
    }
    let mut parts = vec![format!(
        "{}/{} recompressed",
        stats.images_recompressed, stats.total_image_objects
    )];
    if stats.images_kept_original > 0 {
        parts.push(format!("{} kept", stats.images_kept_original));
    }
    if stats.images_skipped_small_or_mask > 0 {
        parts.push(format!("{} small/masks", stats.images_skipped_small_or_mask));
    }
    if stats.images_skipped_unsupported > 0 {
        if let Some(first_reason) = stats.skipped_reasons.first() {
            parts.push(format!(
                "{} skipped ({first_reason})",
                stats.images_skipped_unsupported
            ));
        } else {
            parts.push(format!("{} skipped", stats.images_skipped_unsupported));
        }
    }
    parts.join(", ")
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

    fn make_test_item(name: &str, file_bytes: usize) -> PdfFileItem {
        let img_count = (file_bytes / 60_000).max(10);
        let sample_orig_bytes = file_bytes.saturating_sub(10_000) / img_count;
        let summary = PdfSummary::synthetic_for_test(
            3,
            file_bytes,
            10_000,
            img_count,
            400,
            sample_orig_bytes,
            true,
        );
        let settings = CompressionSettings::for_file(bytes_to_mb(file_bytes) as f32);
        let cached_estimated_bytes = summary.estimate_compressed_bytes(&settings.options);
        PdfFileItem {
            path: Some(PathBuf::from(format!(r"C:\docs\{name}"))),
            file_name: name.to_string(),
            original_size_mb: bytes_to_mb(file_bytes),
            bytes: Arc::new(Vec::new()),
            summary,
            settings,
            cached_estimated_bytes,
            result: None,
            error: None,
        }
    }

    fn run_frame(
        ctx: &egui::Context,
        app: &mut PdfCompressorApp,
        width: f32,
        height: f32,
    ) -> egui::FullOutput {
        run_frame_with_input(ctx, app, width, height, None, true)
    }

    fn run_frame_with_input(
        ctx: &egui::Context,
        app: &mut PdfCompressorApp,
        width: f32,
        height: f32,
        extra_events: Option<Vec<egui::Event>>,
        focused: bool,
    ) -> egui::FullOutput {
        let mut raw_input = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::Pos2::ZERO,
                egui::vec2(width, height),
            )),
            focused,
            ..Default::default()
        };
        if let Some(ev) = extra_events {
            raw_input.events = ev;
        }
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

        // Simulate single-file completed save ("Saved to: ..." appears in Compression Summary)
        let mut item = make_test_item("homework.pdf", 4 * 1024 * 1024);
        item.result = Some(CompressionResult::from_stats(
            PathBuf::from(r"C:\Users\demo\homework_compressed.pdf"),
            CompressionStats {
                original_bytes: 4 * 1024 * 1024,
                compressed_bytes: 1_200_000,
                ..Default::default()
            },
            &item.settings,
            item.original_size_mb,
        ));
        app.files = vec![item];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);

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

        // Narrowest supported width (480px) fits controls cleanly without horizontal overflow
        let mut compact_app = PdfCompressorApp::default();
        let mut compact_item = make_test_item("homework_scan_2026.pdf", 3_500_000);
        compact_item.settings.options.format = ImageFormatChoice::Png;
        compact_item.settings.target_mode = TargetMode::Megabytes;
        compact_app.files = vec![compact_item];
        compact_app.selected_indices.insert(0);
        compact_app.selection_anchor = Some(0);
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
            app.single_summary_metrics(),
            ("-".to_string(), "-".to_string(), "-".to_string(), false)
        );

        // Loaded before compression: inline estimate has no tilde and summary shows muted estimate
        let item = make_test_item("notes.pdf", 4_000_000);
        let inline_est = estimated_size_text(&item);
        assert!(inline_est.starts_with("Estimated Size: "));
        assert!(!inline_est.contains('~'));
        assert!(!inline_est.contains('%'));

        app.files = vec![item];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);
        let (orig, new_sz, saved, is_est) = app.single_summary_metrics();
        assert_ne!(orig, "-");
        assert_eq!(new_sz, "-");
        assert!(saved.starts_with('~') && saved.ends_with('%') && !saved.contains("est."));
        assert!(is_est);
        assert_eq!(ImageFormatChoice::Png.label(), "PNG");

        // After compression: actual percentage replaces muted estimate as bold actual text
        let orig_mb = app.files[0].original_size_mb;
        app.files[0].result = Some(CompressionResult::from_stats(
            PathBuf::from("notes_compressed.pdf"),
            CompressionStats {
                original_bytes: 4_000_000,
                compressed_bytes: 1_156_000,
                ..Default::default()
            },
            &app.files[0].settings,
            orig_mb,
        ));
        let (orig, new_sz, saved, is_est) = app.single_summary_metrics();
        assert_ne!(orig, "-");
        assert_ne!(new_sz, "-");
        assert_eq!(saved, "71.1%");
        assert!(!is_est);
    }

    #[test]
    fn multi_select_click_ctrl_shift_and_ctrl_a_behave_accurately() {
        let mut app = PdfCompressorApp::default();
        app.files = (1..=6)
            .map(|i| make_test_item(&format!("page_{i}.pdf"), i * 1_000_000))
            .collect();
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);

        // Plain click selects only row 2 and updates anchor
        app.handle_row_click(2, egui::Modifiers::NONE);
        assert_eq!(app.selected_indices, BTreeSet::from([2]));
        assert_eq!(app.selection_anchor, Some(2));

        // Shift+click downwards selects contiguous range 2..=4 preserving anchor 2
        app.handle_row_click(4, egui::Modifiers::SHIFT);
        assert_eq!(app.selected_indices, BTreeSet::from([2, 3, 4]));
        assert_eq!(app.selection_anchor, Some(2));

        // Shift+click upwards from anchor 2 selects contiguous range 0..=2
        app.handle_row_click(0, egui::Modifiers::SHIFT);
        assert_eq!(app.selected_indices, BTreeSet::from([0, 1, 2]));
        assert_eq!(app.selection_anchor, Some(2));

        // Ctrl+click toggles individual rows without clearing others and updates anchor
        app.handle_row_click(5, egui::Modifiers::COMMAND);
        assert_eq!(app.selected_indices, BTreeSet::from([0, 1, 2, 5]));
        assert_eq!(app.selection_anchor, Some(5));
        app.handle_row_click(1, egui::Modifiers::COMMAND);
        assert_eq!(app.selected_indices, BTreeSet::from([0, 2, 5]));
        assert_eq!(app.selection_anchor, Some(1));

        // Ctrl+A respects window focus and text-input focus
        let ctx = egui::Context::default();
        let ctrl_a_events = vec![egui::Event::Key {
            key: egui::Key::A,
            physical_key: None,
            pressed: true,
            repeat: false,
            modifiers: egui::Modifiers::COMMAND,
        }];

        // Unfocused window should NOT select all
        let _ = run_frame_with_input(
            &ctx,
            &mut app,
            DEFAULT_WINDOW_WIDTH,
            640.0,
            Some(ctrl_a_events.clone()),
            false,
        );
        assert_eq!(app.selected_indices, BTreeSet::from([0, 2, 5]));

        // Focused window with no keyboard-input widget DOES select all rows
        let _ = run_frame_with_input(
            &ctx,
            &mut app,
            DEFAULT_WINDOW_WIDTH,
            640.0,
            Some(ctrl_a_events),
            true,
        );
        assert_eq!(app.selected_indices, BTreeSet::from([0, 1, 2, 3, 4, 5]));
    }

    #[test]
    fn per_file_vs_apply_for_all_clamping_and_batch_export_paths() {
        let mut app = PdfCompressorApp::default();
        app.files = vec![
            make_test_item("small.pdf", 1_000_000),   // ~0.95 MB
            make_test_item("medium.pdf", 4_000_000),  // ~3.81 MB
            make_test_item("same\\small.pdf", 8_000_000),
        ];
        app.selected_indices = BTreeSet::from([1]);
        app.selection_anchor = Some(1);

        // With Apply for ALL unchecked, changing quality affects only selected index 1
        let mut draft = app.files[1].settings.clone();
        draft.options.quality = 82;
        app.apply_settings_delta(
            &draft,
            SettingsDelta {
                quality_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.options.quality, 100);
        assert_eq!(app.files[1].settings.options.quality, 82);
        assert_eq!(app.files[2].settings.options.quality, 100);

        // Toggling Apply for ALL copies active file settings across all files
        app.apply_for_all = true;
        app.sync_all_from_active();
        assert_eq!(app.files[0].settings.options.quality, 82);
        assert_eq!(app.files[2].settings.options.quality, 82);

        // Enabling Target Size (3.0 MB) in Apply for ALL clamps small files while syncing larger ones
        let mut target_draft = app.files[1].settings.clone();
        target_draft.target_mode = TargetMode::Megabytes;
        target_draft.target_size_mb = 3.0;
        app.apply_settings_delta(
            &target_draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                target_mb_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.target_mode, TargetMode::Megabytes);
        assert_eq!(app.files[2].settings.target_mode, TargetMode::Megabytes);
        assert!(app.files[0].settings.target_size_mb <= app.files[0].original_size_mb as f32 + 1e-5);
        assert!((app.files[2].settings.target_size_mb - 3.0).abs() < 1e-4);

        // Raising Target Size in Apply for ALL updates large files without letting small clamp truncate shared value
        let mut big_target_draft = app.files[1].settings.clone();
        big_target_draft.target_size_mb = 5.0;
        app.apply_settings_delta(
            &big_target_draft,
            SettingsDelta {
                target_mb_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(app.shared_target_size_mb, Some(5.0));
        assert!(app.files[0].settings.target_size_mb <= app.files[0].original_size_mb as f32 + 1e-5);
        assert!(app.files[1].settings.target_size_mb <= app.files[1].original_size_mb as f32 + 1e-5);
        assert!((app.files[2].settings.target_size_mb - 5.0).abs() < 1e-4);

        // Unique batch output paths preserve <stem>_compressed.pdf and disambiguate collisions
        let out_dir = PathBuf::from(r"C:\out");
        let paths = unique_batch_output_paths(&app.files, &out_dir);
        assert_eq!(paths[0], out_dir.join("small_compressed.pdf"));
        assert_eq!(paths[1], out_dir.join("medium_compressed.pdf"));
        assert_ne!(paths[2], paths[0]);
    }

    #[test]
    fn batch_ui_renders_cleanly_at_min_width_and_caps_scroll_for_many_files() {
        let ctx = egui::Context::default();
        let mut app_small_batch = PdfCompressorApp::default();
        app_small_batch.files = (1..=3)
            .map(|i| make_test_item(&format!("doc_{i}.pdf"), (i + 1) * 1_500_000))
            .collect();
        app_small_batch.selected_indices = BTreeSet::from([0]);
        app_small_batch.selection_anchor = Some(0);
        app_small_batch.files[0].settings.target_mode = TargetMode::Megabytes;
        app_small_batch.files[0].settings.target_size_mb = 1.0;
        app_small_batch.files[0].result = Some(CompressionResult::from_stats(
            PathBuf::from(r"C:\out\doc_1_compressed.pdf"),
            CompressionStats {
                original_bytes: 3_000_000,
                compressed_bytes: 1_500_000,
                ..Default::default()
            },
            &app_small_batch.files[0].settings,
            app_small_batch.files[0].original_size_mb,
        ));
        assert!(app_small_batch.files[0].result.as_ref().unwrap().over_target);

        app_small_batch.files[1].result = Some(CompressionResult::from_stats(
            PathBuf::from(r"C:\out\doc_2_compressed.pdf"),
            CompressionStats {
                original_bytes: 4_500_000,
                compressed_bytes: 1_200_000,
                ..Default::default()
            },
            &app_small_batch.files[1].settings,
            app_small_batch.files[1].original_size_mb,
        ));
        assert!(!app_small_batch.files[1].result.as_ref().unwrap().over_target);
        app_small_batch.batch_saved_folder = Some(PathBuf::from(r"C:\out"));

        let _ = run_frame(&ctx, &mut app_small_batch, MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app_small_batch.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "multi-file UI overflowed {MIN_WINDOW_WIDTH}px width: {}px",
            app_small_batch.last_content_width
        );
        let three_file_h = app_small_batch.last_fitted_height.unwrap();

        let mut app_large_batch = PdfCompressorApp::default();
        app_large_batch.files = (1..=15)
            .map(|i| make_test_item(&format!("scan_{i:02}.pdf"), 2_500_000))
            .collect();
        app_large_batch.selected_indices = BTreeSet::from([0]);
        app_large_batch.selection_anchor = Some(0);
        let _ = run_frame(&ctx, &mut app_large_batch, DEFAULT_WINDOW_WIDTH, 900.0);
        let fifteen_file_h = app_large_batch.last_fitted_height.unwrap();
        assert!(
            fifteen_file_h > three_file_h && fifteen_file_h < 900.0,
            "15-file batch height ({fifteen_file_h}) should expand cleanly up to capped visible rows"
        );
    }

    #[test]
    fn table_columns_align_deterministically_even_with_long_filenames() {
        let total_w = 430.0;
        let file_w = file_table_col_widths(total_w);
        assert_eq!(file_w[0], 40.0);
        assert_eq!(file_w[1], total_w - 40.0 - 85.0 - 85.0);
        assert_eq!(file_w[2], 85.0);
        assert_eq!(file_w[3], 85.0);

        let sum_w = summary_table_col_widths(total_w);
        let expected_third = (total_w - 44.0) / 3.0;
        assert_eq!(sum_w[0], 44.0);
        assert!((sum_w[1] - expected_third).abs() < 1e-5);
        assert!((sum_w[2] - expected_third).abs() < 1e-5);
        assert!((sum_w[3] - expected_third).abs() < 1e-5);

        let row_rect = egui::Rect::from_min_size(
            egui::pos2(12.0, 20.0),
            egui::vec2(total_w, TABLE_ROW_HEIGHT),
        );
        let cols = split_row_columns(row_rect, file_w);
        assert_eq!(cols[0].min.x, row_rect.min.x);
        assert_eq!(cols[0].max.x, cols[1].min.x);
        assert_eq!(cols[1].max.x, cols[2].min.x);
        assert_eq!(cols[2].max.x, cols[3].min.x);
        assert_eq!(cols[3].max.x, row_rect.max.x);

        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();
        app.files = vec![
            make_test_item("short.pdf", 1_200_000),
            make_test_item(
                "very_long_scanned_homework_document_filename_2026_semester_final_submission_v3.pdf",
                2_400_000,
            ),
        ];
        app.selected_indices = BTreeSet::from([0]);
        app.selection_anchor = Some(0);

        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "long filenames should truncate instead of overflowing {MIN_WINDOW_WIDTH}px: {}px",
            app.last_content_width
        );
    }

    #[test]
    fn file_est_column_formats_mb_and_percent_of_original_cleanly() {
        assert_eq!(format_file_est_cell(3_000_000, 1_020_000, false), "0.97 MB");
        // Percent mode shows remaining ratio (est / orig * 100), not space saved (66%).
        assert_eq!(format_file_est_cell(3_000_000, 1_020_000, true), "34%");
        assert_eq!(format_file_est_cell(2_000_000, 2_000_000, true), "100%");
        assert_eq!(format_file_est_cell(0, 100, true), "-");

        // Verify JPG/JPEG 2000/PNG render with identical controls height (no compatibility warning banner).
        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();
        app.files = vec![make_test_item("one.pdf", 2_000_000)];
        app.selected_indices = BTreeSet::from([0]);
        app.selection_anchor = Some(0);

        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        let jpeg_h = app.last_fitted_height.unwrap();

        app.files[0].settings.options.format = ImageFormatChoice::Jpeg2000;
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        let jpx_h = app.last_fitted_height.unwrap();

        app.files[0].settings.options.format = ImageFormatChoice::Png;
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        let png_h = app.last_fitted_height.unwrap();

        assert!((jpeg_h - jpx_h).abs() < 1.0);
        assert!((jpeg_h - png_h).abs() < 1.0);
    }

    #[test]
    fn target_mode_cycles_three_states_and_couples_percent_target_cleanly() {
        // Sequential 3-state transition: Off -> Megabytes -> Percent -> Off
        assert_eq!(TargetMode::Off.next(), TargetMode::Megabytes);
        assert_eq!(TargetMode::Megabytes.next(), TargetMode::Percent);
        assert_eq!(TargetMode::Percent.next(), TargetMode::Off);
        assert!(!TargetMode::Off.is_enabled());
        assert!(TargetMode::Megabytes.is_enabled());
        assert!(TargetMode::Percent.is_enabled());

        // Percentage initializer derives cleanly from MB target or falls back to 100%
        assert_eq!(initial_target_pct_from_mb(1.2, 4.0), 30.0);
        assert_eq!(initial_target_pct_from_mb(0.0, 4.0), 100.0);
        assert_eq!(initial_target_pct_from_mb(6.0, 4.0), 100.0);

        let mut app = PdfCompressorApp::default();
        let first = make_test_item("first.pdf", 4_000_000); // ~3.81 MB
        let second = make_test_item("second.pdf", 8_000_000); // ~7.63 MB
        app.files = vec![first, second];
        app.selected_indices = BTreeSet::from([0]);
        app.selection_anchor = Some(0);
        app.apply_for_all = true;

        // Switching to Percent (30%) computes per-file effective MB targets and solves inverse scale
        let mut draft = app.files[0].settings.clone();
        draft.target_mode = TargetMode::Percent;
        draft.target_size_pct = 30.0;
        app.apply_settings_delta(
            &draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                target_pct_changed: true,
                ..Default::default()
            },
        );

        let eff0 = app.files[0]
            .settings
            .effective_target_mb(app.files[0].original_size_mb)
            .unwrap();
        let eff1 = app.files[1]
            .settings
            .effective_target_mb(app.files[1].original_size_mb)
            .unwrap();
        assert!((eff0 - app.files[0].original_size_mb * 0.30).abs() < 1e-6);
        assert!((eff1 - app.files[1].original_size_mb * 0.30).abs() < 1e-6);

        // Changing Quality in Percent mode automatically solves Resolution Scale to fit 30% target
        let mut quality_draft = app.files[0].settings.clone();
        quality_draft.options.quality = 85;
        app.apply_settings_delta(
            &quality_draft,
            SettingsDelta {
                quality_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.options.quality, 85);
        assert!(app.files[0].settings.options.scale < 0.99);

        // Changing Resolution Scale in Percent mode solves Quality
        let mut scale_draft = app.files[0].settings.clone();
        scale_draft.options.scale = 1.0;
        app.apply_settings_delta(
            &scale_draft,
            SettingsDelta {
                scale_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.options.scale, 1.0);
        assert!(app.files[0].settings.options.quality < 85);

        // Summary comparison uses effective_target_mb = original_size_mb * (pct / 100.0)
        // first.pdf is 4_000_000 B -> 30% target is 1_200_000 B
        let under = CompressionResult::from_stats(
            PathBuf::from(r"C:\out\first_ok.pdf"),
            CompressionStats {
                original_bytes: 4_000_000,
                compressed_bytes: 1_100_000,
                ..Default::default()
            },
            &app.files[0].settings,
            app.files[0].original_size_mb,
        );
        assert!(!under.over_target);

        let over = CompressionResult::from_stats(
            PathBuf::from(r"C:\out\first_over.pdf"),
            CompressionStats {
                original_bytes: 4_000_000,
                compressed_bytes: 1_350_000,
                ..Default::default()
            },
            &app.files[0].settings,
            app.files[0].original_size_mb,
        );
        assert!(over.over_target);

        // Over-target single summary card renders only badge + red highlight (zero warning banner text height).
        let ctx = egui::Context::default();
        app.files[0].result = Some(under);
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        let under_h = app.last_fitted_height.unwrap();

        app.files[0].result = Some(over);
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        let over_h = app.last_fitted_height.unwrap();
        assert!(
            (under_h - over_h).abs() < 1.0,
            "Over-target summary should have no extra warning text height ({under_h} vs {over_h})"
        );
        app.files[0].result = None;

        // Percent mode UI renders cleanly at MIN_WINDOW_WIDTH without horizontal overflow
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "Percent mode UI overflowed {MIN_WINDOW_WIDTH}px width: {}px",
            app.last_content_width
        );
    }

    #[test]
    fn load_resets_defaults_to_q100_100x_off_and_surfaces_encrypted_or_empty_errors() {
        use lopdf::{dictionary, Document, Object, Stream};

        let mut app = PdfCompressorApp::default();
        app.unloaded_settings.options.quality = 33;
        app.unloaded_settings.options.scale = 0.25;
        app.unloaded_settings.target_mode = TargetMode::Megabytes;

        let mut valid_doc = Document::with_version("1.4");
        let p_root = valid_doc.new_object_id();
        let cont_id = valid_doc.add_object(Stream::new(dictionary! {}, b"BT ET".to_vec()));
        let page_id = valid_doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => p_root,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => cont_id,
        });
        valid_doc.objects.insert(
            p_root,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 1,
                "Kids" => vec![Object::Reference(page_id)],
            }),
        );
        let c_id = valid_doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => p_root });
        valid_doc.trailer.set("Root", c_id);
        let mut valid_bytes = Vec::new();
        valid_doc.save_to(&mut valid_bytes).expect("save valid");

        let mut enc_doc = valid_doc.clone();
        let enc_id = enc_doc.add_object(dictionary! { "Filter" => "Standard", "V" => 2 });
        enc_doc.trailer.set("Encrypt", enc_id);
        let mut enc_bytes = Vec::new();
        enc_doc.save_to(&mut enc_bytes).expect("save enc");

        app.load_pdf_inputs(
            vec![
                (None, "ok.pdf".to_string(), valid_bytes),
                (None, "empty.pdf".to_string(), Vec::new()),
                (None, "locked.pdf".to_string(), enc_bytes),
            ],
            Vec::new(),
        );

        assert_eq!(app.files.len(), 1);
        assert_eq!(app.files[0].settings.options.quality, 100);
        assert!((app.files[0].settings.options.scale - 1.00).abs() < 1e-6);
        assert_eq!(app.files[0].settings.target_mode, TargetMode::Off);
        assert!((app.files[0].settings.target_size_mb - (app.files[0].original_size_mb as f32)).abs() < 1e-6);
        assert!((app.files[0].settings.target_size_pct - 100.0).abs() < 1e-6);
        assert_eq!(app.files[0].settings.options.format, ImageFormatChoice::Jpeg);
        assert_eq!(
            app.files[0].settings.options.resize_filter,
            ResizeFilterChoice::Lanczos3
        );
        assert_eq!(app.unloaded_settings.options.quality, 100);
        assert_eq!(app.unloaded_settings.target_mode, TargetMode::Off);

        let err = app.error_message.as_deref().expect("errors surfaced");
        assert!(err.contains("empty.pdf"), "missing empty error in: {err}");
        assert!(err.contains("locked.pdf"), "missing encrypted error in: {err}");
    }

    #[test]
    fn fine_scale_step_and_default_targets_preserve_q100_100x_until_lowered() {
        let mut app = PdfCompressorApp::default();
        let first = make_test_item("one.pdf", 4_000_000);
        let second = make_test_item("two.pdf", 12_000_000);

        // Fresh items initialize Target Size defaults to exact original_size_mb and 100%.
        assert!(
            (first.settings.target_size_mb - (first.original_size_mb as f32)).abs() < 1e-6,
            "expected first target_size_mb {}, got {}",
            first.original_size_mb,
            first.settings.target_size_mb
        );
        assert!((first.settings.target_size_pct - 100.0).abs() < 1e-6);
        assert!(
            (second.settings.target_size_mb - (second.original_size_mb as f32)).abs() < 1e-6,
            "expected second target_size_mb {}, got {}",
            second.original_size_mb,
            second.settings.target_size_mb
        );
        assert!((second.settings.target_size_pct - 100.0).abs() < 1e-6);

        app.files = vec![first, second];
        app.selected_indices = BTreeSet::from([0]);
        app.selection_anchor = Some(0);

        // Fine-grained 0.01x scale values (e.g., 0.63x, 0.82x) are preserved without 0.05x snapping.
        let mut scale_draft = app.files[0].settings.clone();
        scale_draft.options.scale = 0.63;
        app.apply_settings_delta(
            &scale_draft,
            SettingsDelta {
                scale_changed: true,
                ..Default::default()
            },
        );
        assert!((app.files[0].settings.options.scale - 0.63).abs() < 1e-6);

        scale_draft.options.scale = 0.82;
        app.apply_settings_delta(
            &scale_draft,
            SettingsDelta {
                scale_changed: true,
                ..Default::default()
            },
        );
        assert!((app.files[0].settings.options.scale - 0.82).abs() < 1e-6);

        // Reset back to default Q100 / 1.00x, then cycle Off -> MB -> % -> Off without premature downscaling.
        app.files[0].settings.options.scale = 1.00;
        app.files[0].settings.options.quality = 100;

        // Click 1: Off -> Target Size (MB) at original_size_mb
        let mut mb_draft = app.files[0].settings.clone();
        mb_draft.target_mode = TargetMode::Megabytes;
        app.apply_settings_delta(
            &mb_draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.target_mode, TargetMode::Megabytes);
        assert!(
            (app.files[0].settings.target_size_mb - (app.files[0].original_size_mb as f32)).abs()
                < 1e-6
        );
        assert_eq!(app.files[0].settings.options.quality, 100);
        assert!((app.files[0].settings.options.scale - 1.00).abs() < 1e-6);

        // Click 2: Target Size (MB) -> Target Size (%) starts at 100% and still keeps Q100 / 1.00x
        let mut pct_draft = app.files[0].settings.clone();
        pct_draft.target_mode = TargetMode::Percent;
        pct_draft.target_size_pct = initial_target_pct_from_mb(
            pct_draft.target_size_mb,
            app.files[0].original_size_mb,
        );
        assert!((pct_draft.target_size_pct - 100.0).abs() < 1e-6);
        app.apply_settings_delta(
            &pct_draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                ..Default::default()
            },
        );
        assert_eq!(app.files[0].settings.target_mode, TargetMode::Percent);
        assert!((app.files[0].settings.target_size_pct - 100.0).abs() < 1e-6);
        assert_eq!(app.files[0].settings.options.quality, 100);
        assert!((app.files[0].settings.options.scale - 1.00).abs() < 1e-6);

        // Lowering Target Size below 100% DOES engage the inverse solver to 0.01x precision.
        let mut lowered_draft = app.files[0].settings.clone();
        lowered_draft.target_size_pct = 50.0;
        app.apply_settings_delta(
            &lowered_draft,
            SettingsDelta {
                target_pct_changed: true,
                ..Default::default()
            },
        );
        let solved = app.files[0].settings.options.scale;
        assert!(solved < 0.99, "expected scale reduction at 50% target, got {solved}");
        assert_ne!(
            ((solved * 100.0).round() as i32) % 5,
            0,
            "expected solve_scale_for_target to land on a 0.01 step not divisible by 0.05, got {solved}"
        );

        // Solver searches 0.01x increments (not coarse 0.05x multiples) when Quality is dragged.
        let mut q92_draft = app.files[0].settings.clone();
        q92_draft.options.quality = 92;
        app.apply_settings_delta(
            &q92_draft,
            SettingsDelta {
                quality_changed: true,
                ..Default::default()
            },
        );
        let q92_scale = app.files[0].settings.options.scale;
        assert!(
            q92_scale > solved && q92_scale < 1.0,
            "expected finer scale between {solved} and 1.0 at Q92, got {q92_scale}"
        );

        // UI rendering preserves fine 0.01x hundredth values (e.g., 0.57x) without snapping to 0.05x.
        app.files[0].settings.target_mode = TargetMode::Off;
        app.files[0].settings.options.scale = 0.57;
        let ctx = egui::Context::default();
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, 700.0);
        assert!((app.files[0].settings.options.scale - 0.57).abs() < 1e-6);
    }

    #[test]
    fn background_load_populates_files_asynchronously_without_blocking_ui() {
        use lopdf::{dictionary, Document, Object, Stream};

        let mut doc = Document::with_version("1.4");
        let p_root = doc.new_object_id();
        let cont_id = doc.add_object(Stream::new(dictionary! {}, b"BT ET".to_vec()));
        let page_id = doc.add_object(dictionary! {
            "Type" => "Page",
            "Parent" => p_root,
            "MediaBox" => vec![0.into(), 0.into(), 612.into(), 792.into()],
            "Contents" => cont_id,
        });
        doc.objects.insert(
            p_root,
            Object::Dictionary(dictionary! {
                "Type" => "Pages",
                "Count" => 1,
                "Kids" => vec![Object::Reference(page_id)],
            }),
        );
        let c_id = doc.add_object(dictionary! { "Type" => "Catalog", "Pages" => p_root });
        doc.trailer.set("Root", c_id);
        let mut bytes = Vec::new();
        doc.save_to(&mut bytes).expect("save async test pdf");

        let mut app = PdfCompressorApp::default();
        let ctx = egui::Context::default();
        app.start_loading_job(
            Vec::new(),
            vec![(None, "async_doc.pdf".to_string(), bytes)],
            Vec::new(),
            &ctx,
        );
        assert!(app.is_busy(), "UI should show busy state immediately without blocking");
        assert!(app.files.is_empty());

        let start = std::time::Instant::now();
        while app.is_busy() && start.elapsed() < std::time::Duration::from_secs(5) {
            app.poll_worker();
            std::thread::sleep(std::time::Duration::from_millis(5));
        }

        assert!(!app.is_busy(), "background load should finish cleanly");
        assert_eq!(app.files.len(), 1);
        assert_eq!(app.files[0].file_name, "async_doc.pdf");
        assert_eq!(app.files[0].settings.options.quality, 100);
        assert!((app.files[0].settings.options.scale - 1.00).abs() < 1e-6);
        assert_eq!(app.files[0].settings.target_mode, TargetMode::Off);
    }

    #[test]
    fn instant_multi_file_target_modes_and_quality_scale_sync_all_rows_immediately() {
        let mut app = PdfCompressorApp::default();
        let a = make_test_item("a.pdf", 4_000_000);
        let b = make_test_item("b.pdf", 10_000_000);
        let c = make_test_item("c.pdf", 20_000_000);
        app.files = vec![a, b, c];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);
        app.apply_for_all = true;

        // 1. Switching directly to Target Size (%) updates every file individually & synchronously.
        let mut pct_draft = app.files[0].settings.clone();
        pct_draft.target_mode = TargetMode::Percent;
        pct_draft.target_size_pct = 45.0;
        app.apply_settings_delta(
            &pct_draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                target_pct_changed: true,
                ..Default::default()
            },
        );
        for item in &app.files {
            let target_limit = mb_to_bytes(item.original_size_mb * 0.45);
            assert_eq!(item.settings.target_mode, TargetMode::Percent);
            assert!((item.settings.target_size_pct - 45.0).abs() < 1e-4);
            assert!(
                item.estimated_compressed_bytes() <= target_limit,
                "{}: estimate {} should synchronously fit 45% limit {}",
                item.file_name,
                item.estimated_compressed_bytes(),
                target_limit
            );
        }

        // 2. Updating Quality Factor recalculates Resolution Scale & estimates synchronously across all files.
        let prev_ests: Vec<usize> = app
            .files
            .iter()
            .map(PdfFileItem::estimated_compressed_bytes)
            .collect();
        let mut q_draft = app.files[0].settings.clone();
        q_draft.options.quality = 72;
        app.apply_settings_delta(
            &q_draft,
            SettingsDelta {
                quality_changed: true,
                ..Default::default()
            },
        );
        for (idx, item) in app.files.iter().enumerate() {
            assert_eq!(item.settings.options.quality, 72);
            assert!(item.settings.options.scale < 1.00);
            assert!(
                item.estimated_compressed_bytes() <= mb_to_bytes(item.original_size_mb * 0.45),
                "{} estimate should stay within 45% target after Quality Factor change (prev={})",
                item.file_name,
                prev_ests[idx]
            );
        }

        // 3. Switching to Target Size (MB) with Apply for ALL clamps per-file to original_size_mb and solves.
        let mut mb_draft = app.files[0].settings.clone();
        mb_draft.target_mode = TargetMode::Megabytes;
        mb_draft.target_size_mb = 2.0;
        app.apply_settings_delta(
            &mb_draft,
            SettingsDelta {
                target_mode_changed: true,
                target_mode_activated: true,
                target_mb_changed: true,
                ..Default::default()
            },
        );
        for item in &app.files {
            assert_eq!(item.settings.target_mode, TargetMode::Megabytes);
            assert!((item.settings.target_size_mb - 2.0).abs() < 1e-4);
            let limit = mb_to_bytes(2.0);
            assert!(item.estimated_compressed_bytes() <= limit);
        }
    }

    #[test]
    fn summary_header_progress_counter_and_slider_alignment_render_cleanly() {
        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();
        app.files = vec![make_test_item("single.pdf", 4_000_000)];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);

        // Two frames let egui::Grid settle its shared column width cleanly at MIN_WINDOW_WIDTH.
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);

        let q_slider = app.last_quality_slider_rect.expect("quality slider rect");
        let s_slider = app.last_scale_slider_rect.expect("scale slider rect");
        let q_drag = app.last_quality_drag_rect.expect("quality drag rect");
        let s_drag = app.last_scale_drag_rect.expect("scale drag rect");
        assert!(
            (q_slider.min.x - s_slider.min.x).abs() < 0.5
                && (q_slider.width() - s_slider.width()).abs() < 0.5
                && (q_drag.max.x - s_drag.max.x).abs() < 0.5
                && (q_drag.width() - s_drag.width()).abs() < 0.5,
            "sliders and value boxes should align identically: q={q_slider:?}/{q_drag:?}, s={s_slider:?}/{s_drag:?}"
        );
        assert!(app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5);

        // Busy state renders cleanly with compact progress counter (single 1/1 and batch 2/3)
        let (_tx, rx) = mpsc::channel();
        app.worker_rx = Some(rx);
        app.progress = Some(ProgressUpdate {
            current: 1,
            total: 1,
        });
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        assert!(app.is_busy());

        app.files = vec![
            make_test_item("one.pdf", 2_000_000),
            make_test_item("two.pdf", 3_000_000),
            make_test_item("three.pdf", 4_000_000),
        ];
        app.progress = Some(ProgressUpdate {
            current: 2,
            total: 3,
        });
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, 700.0);
        assert!(app.is_busy());

        app.worker_rx = None;
        app.progress = None;
        let _ = run_frame(&ctx, &mut app, MIN_WINDOW_WIDTH, 700.0);
        assert!(!app.is_busy());
    }
}
