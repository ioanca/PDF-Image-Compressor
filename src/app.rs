use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};

use eframe::egui::{self, Color32, CornerRadius, Frame, Margin, RichText, Stroke};
use smallvec::SmallVec;
use rfd::FileDialog;

use crate::pdf_processor::{
    bytes_to_mb, compress_pdf, compress_pdf_path, inspect_pdf, inspect_pdf_path, mb_to_bytes,
    render_page_preview_cached, set_current_thread_below_normal, ColorDepth, CompressionOptions,
    CompressionStats, ImageFormatChoice, PagePreviewData, PdfSummary, PreviewSessionCache,
    PreviewSource, ProgressUpdate, ResizeFilterChoice,
};

fn next_file_source_id() -> u64 {
    static NEXT_SOURCE_ID: AtomicU64 = AtomicU64::new(1);
    NEXT_SOURCE_ID.fetch_add(1, Ordering::Relaxed)
}

const WORKER_STACK_BYTES: usize = 8 * 1024 * 1024;

#[inline]
fn spawn_worker_thread<F>(name: &str, f: F) -> bool
where
    F: FnOnce() + Send + 'static,
{
    spawn_worker_thread_boxed(name, Box::new(f))
}

fn spawn_worker_thread_boxed(name: &str, f: Box<dyn FnOnce() + Send + 'static>) -> bool {
    thread::Builder::new()
        .name(name.to_string())
        .stack_size(WORKER_STACK_BYTES)
        .spawn(f)
        .is_ok()
}

pub const DEFAULT_WINDOW_WIDTH: f32 = 528.0;
pub const DEFAULT_WINDOW_HEIGHT: f32 = 520.0;
pub const MIN_WINDOW_WIDTH: f32 = 528.0;
pub const MIN_WINDOW_HEIGHT: f32 = 520.0;
#[cfg(test)]
pub const PREVIEW_DEFAULT_WIDTH: f32 = 816.0;
pub const PREVIEW_MIN_WIDTH: f32 = 320.0;
#[cfg(test)]
const MIN_PREVIEW_WINDOW_HEIGHT: f32 = 520.0;
const PREVIEW_DEBOUNCE_MS: u64 = 120;

fn dpi_scaled_widths(ctx: &egui::Context) -> (f32, f32) {
    let monitor_w_pts = ctx
        .input(|i| i.viewport().monitor_size.map(|m| m.x))
        .filter(|w| w.is_finite() && *w > 0.0)
        .unwrap_or_else(|| 1920.0 / ctx.pixels_per_point().max(1.0));
    let main_width = monitor_w_pts * 0.275;
    let default_preview_width = monitor_w_pts * 0.425;
    (main_width, default_preview_width)
}

const FILE_TABLE_MAX_VISIBLE_ROWS: usize = 10;
const SUMMARY_TABLE_MIN_VISIBLE_ROWS: usize = 5;
const SUMMARY_TABLE_MAX_VISIBLE_ROWS: usize = 10;

fn summary_rows_height(rows: usize) -> f32 {
    if rows == 0 {
        0.0
    } else {
        (rows as f32) * TABLE_ROW_HEIGHT + (rows.saturating_sub(1) as f32) * TABLE_ROW_GAP
    }
}
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
    suffix: String,
    target_mode: TargetMode,
    target_size_mb: f32,
    target_size_pct: f32,
}

impl Default for CompressionSettings {
    fn default() -> Self {
        Self {
            options: CompressionOptions::default(),
            suffix: "-compressed".to_string(),
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
    source_id: u64,
    path: Option<PathBuf>,
    file_name: String,
    original_size_mb: f64,
    in_memory_bytes: Option<Arc<Vec<u8>>>,
    summary: PdfSummary,
    settings: CompressionSettings,
    cached_estimated_bytes: usize,
    result: Option<Box<CompressionResult>>,
    error: Option<String>,
}

/// Tightly packed SoA numeric metrics kept separate from cold UI strings/paths so hot
/// batch sizing/estimation scans iterate contiguous primitive arrays.
#[derive(Default)]
struct BatchHotMetrics {
    orig_bytes: Vec<usize>,
    compressible_bytes: Vec<usize>,
    fixed_bytes: Vec<usize>,
    pixel_area: Vec<u64>,
    orig_mb: Vec<f32>,
}

impl BatchHotMetrics {
    fn sync_from(&mut self, files: &[PdfFileItem]) {
        let len = files.len();
        self.orig_bytes.clear();
        self.compressible_bytes.clear();
        self.fixed_bytes.clear();
        self.pixel_area.clear();
        self.orig_mb.clear();
        self.orig_bytes.reserve(len);
        self.compressible_bytes.reserve(len);
        self.fixed_bytes.reserve(len);
        self.pixel_area.reserve(len);
        self.orig_mb.reserve(len);
        files.iter().for_each(|f| {
            let orig = f.summary.file_bytes;
            let fixed = f.summary.fixed_bytes.min(orig);
            self.orig_bytes.push(orig);
            self.fixed_bytes.push(fixed);
            self.compressible_bytes.push(orig.saturating_sub(fixed));
            self.pixel_area.push(
                f.summary
                    .compressible_rgb_pixels
                    .saturating_add(f.summary.compressible_gray_pixels),
            );
            self.orig_mb.push(f.original_size_mb as f32);
        });
    }

    fn max_orig_mb(&self) -> f32 {
        self.orig_mb
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
            .max(1e-6)
    }
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
    path: Option<PathBuf>,
    in_memory_bytes: Option<Arc<Vec<u8>>>,
    original_size_mb: f64,
    settings: CompressionSettings,
    output_path: PathBuf,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewFraming {
    Fit,
    Fill,
    Center,
}

impl PreviewFraming {
    fn label(self) -> &'static str {
        match self {
            Self::Fit => "Fit",
            Self::Fill => "Fill",
            Self::Center => "Center",
        }
    }

    fn next(self) -> Self {
        match self {
            Self::Fit => Self::Fill,
            Self::Fill => Self::Center,
            Self::Center => Self::Fit,
        }
    }

    fn tooltip(self) -> &'static str {
        match self {
            Self::Fit => "Fit entire page inside pane (click to cycle: Fit → Fill → Center)",
            Self::Fill => "Fill pane uniformly with page (click to cycle: Fit → Fill → Center)",
            Self::Center => "Center page at 1.0× reference scale (click to cycle: Fit → Fill → Center)",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PreviewViewMode {
    Framing(PreviewFraming),
    FixedZoomPct(u16),
}

#[derive(Clone, Debug, PartialEq)]
struct PreviewKey {
    source_id: u64,
    file_index: usize,
    file_name: String,
    path: Option<PathBuf>,
    page_number: usize,
    format: ImageFormatChoice,
    color_depth: ColorDepth,
    quality: u8,
    scale_bits: u32,
    resize_filter: ResizeFilterChoice,
}

impl PreviewKey {
    fn from_item(file_index: usize, item: &PdfFileItem, page_number: usize) -> Self {
        Self {
            source_id: item.source_id,
            file_index,
            file_name: item.file_name.clone(),
            path: item.path.clone(),
            page_number,
            format: item.settings.options.format,
            color_depth: item.settings.options.color_depth,
            quality: item.settings.options.quality,
            scale_bits: item.settings.options.scale.to_bits(),
            resize_filter: item.settings.options.resize_filter,
        }
    }

    fn same_source_page(&self, other: &Self) -> bool {
        self.source_id == other.source_id
            && self.file_index == other.file_index
            && self.file_name == other.file_name
            && self.path == other.path
            && self.page_number == other.page_number
    }
}

struct PreviewDisplay {
    key: PreviewKey,
    orig_texture: Option<egui::TextureHandle>,
    orig_width: u32,
    orig_height: u32,
    comp_texture: Option<egui::TextureHandle>,
    comp_width: u32,
    comp_height: u32,
    kept_original: bool,
    synthetic_empty: bool,
    error: Option<String>,
}

struct PreviewWorkerEvent {
    generation: u64,
    key: PreviewKey,
    outcome: Result<Option<Box<PagePreviewData>>, String>,
}

struct PreviewRenderRequest {
    generation: u64,
    key: PreviewKey,
    path: Option<PathBuf>,
    in_memory_bytes: Option<Arc<Vec<u8>>>,
    options: CompressionOptions,
    reply_ctx: egui::Context,
}

enum PreviewCommand {
    Render(Box<PreviewRenderRequest>),
    ClearCache,
}

enum WorkerEvent {
    Progress(ProgressUpdate),
    LoadFinished {
        items: Vec<PdfFileItem>,
        errors: Vec<String>,
    },
    BatchItemFinished {
        index: usize,
        outcome: Result<Box<CompressionResult>, String>,
    },
    BatchFinished {
        folder: PathBuf,
    },
}

#[derive(Default, Clone, Copy)]
struct SettingsDelta {
    format_changed: bool,
    color_depth_changed: bool,
    quality_changed: bool,
    scale_changed: bool,
    filter_changed: bool,
    suffix_changed: bool,
    target_mode_changed: bool,
    target_mode_activated: bool,
    target_mb_changed: bool,
    target_pct_changed: bool,
}

impl SettingsDelta {
    fn any_changed(self) -> bool {
        self.format_changed
            || self.color_depth_changed
            || self.quality_changed
            || self.scale_changed
            || self.filter_changed
            || self.suffix_changed
            || self.target_mode_changed
            || self.target_mb_changed
            || self.target_pct_changed
    }
}

pub struct PdfCompressorApp {
    files: Vec<PdfFileItem>,
    hot_metrics: BatchHotMetrics,
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
    show_preview: bool,
    preview_page: usize,
    preview_page_edit_buf: String,
    preview_framing: PreviewFraming,
    preview_fixed_zoom: bool,
    preview_zoom_pct: u16,
    preview_zoom_exact: f32,
    preview_zoom_editing: bool,
    preview_zoom_just_started: bool,
    preview_zoom_edit_buf: String,
    preview_show_original_held: bool,
    preview_scroll_offset: egui::Vec2,
    preview_display: Option<Box<PreviewDisplay>>,
    preview_pending_key: Option<PreviewKey>,
    preview_pending_since: Option<Instant>,
    preview_inflight_key: Option<PreviewKey>,
    preview_generation: Arc<AtomicU64>,
    preview_cmd_tx: Option<Sender<PreviewCommand>>,
    preview_rx: Option<Receiver<PreviewWorkerEvent>>,
    preview_panel_open: bool,
    last_viewport_lock_state: Option<(bool, u32, u32)>,
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
            hot_metrics: BatchHotMetrics::default(),
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
            show_preview: false,
            preview_page: 1,
            preview_page_edit_buf: "1".to_string(),
            preview_framing: PreviewFraming::Fit,
            preview_fixed_zoom: false,
            preview_zoom_pct: 100,
            preview_zoom_exact: 100.0,
            preview_zoom_editing: false,
            preview_zoom_just_started: false,
            preview_zoom_edit_buf: "100".to_string(),
            preview_show_original_held: false,
            preview_scroll_offset: egui::Vec2::ZERO,
            preview_display: None,
            preview_pending_key: None,
            preview_pending_since: None,
            preview_inflight_key: None,
            preview_generation: Arc::new(AtomicU64::new(0)),
            preview_cmd_tx: None,
            preview_rx: None,
            preview_panel_open: false,
            last_viewport_lock_state: None,
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

    fn clear_preview_textures(&mut self) {
        self.preview_generation.fetch_add(1, Ordering::Relaxed);
        if let Some(tx) = &self.preview_cmd_tx {
            let _ = tx.send(PreviewCommand::ClearCache);
        }
        self.preview_display = None;
        self.preview_pending_key = None;
        self.preview_pending_since = None;
        self.preview_inflight_key = None;
        self.preview_show_original_held = false;
    }

    fn reset_preview_on_file_switch(&mut self) {
        self.preview_page = 1;
        self.preview_page_edit_buf = "1".to_string();
        self.clear_preview_textures();
    }

    fn clear_files(&mut self) {
        self.files.clear();
        self.hot_metrics.sync_from(&self.files);
        self.selected_indices.clear();
        self.selection_anchor = None;
        self.apply_for_all = false;
        self.shared_target_size_mb = None;
        self.batch_saved_folder = None;
        self.status_message = None;
        self.error_message = None;
        self.reset_preview_on_file_switch();
    }

    fn build_loaded_item(
        path: Option<PathBuf>,
        file_name: String,
        in_memory_bytes: Option<Arc<Vec<u8>>>,
        summary: PdfSummary,
    ) -> PdfFileItem {
        let original_size_mb = bytes_to_mb(summary.file_bytes);
        let settings = CompressionSettings::for_file(original_size_mb as f32);
        let cached_estimated_bytes = summary.estimate_compressed_bytes(&settings.options);
        PdfFileItem {
            source_id: next_file_source_id(),
            path,
            file_name,
            original_size_mb,
            in_memory_bytes,
            summary,
            settings,
            cached_estimated_bytes,
            result: None,
            error: None,
        }
    }

    #[cfg(test)]
    fn inspect_loaded_inputs(
        inputs: Vec<(Option<PathBuf>, String, Vec<u8>)>,
        mut errors: Vec<String>,
        on_item: &(dyn Fn() + Sync),
    ) -> (Vec<PdfFileItem>, Vec<String>) {
        let mut loaded_items = Vec::with_capacity(inputs.len());
        for (path, file_name, bytes) in inputs {
            let summary_res = inspect_pdf(&bytes);
            on_item();
            match summary_res {
                Ok(summary) => {
                    loaded_items.push(Self::build_loaded_item(
                        path,
                        file_name,
                        Some(Arc::new(bytes)),
                        summary,
                    ));
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
            self.hot_metrics.sync_from(&self.files);
            self.selected_indices.clear();
            self.selected_indices.insert(0);
            self.selection_anchor = Some(0);
            self.batch_saved_folder = None;
            self.status_message = None;
            self.shared_target_size_mb = None;
            self.unloaded_settings = CompressionSettings::default();
            self.reset_preview_on_file_switch();
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
        inputs: Vec<(Option<PathBuf>, String, Vec<u8>)>,
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
        if !spawn_worker_thread("pdf_loader_worker", move || {
            set_current_thread_below_normal();
            let mut items = Vec::with_capacity(paths.len().saturating_add(inputs.len()));
            let mut done: usize = 0;

            // Parse disk-backed PDFs strictly one at a time without retaining raw bytes or Document.
            for path in paths {
                let file_name = path
                    .file_name()
                    .map(|n| n.to_string_lossy().into_owned())
                    .unwrap_or_else(|| "document.pdf".to_string());
                match inspect_pdf_path(&path) {
                    Ok(summary) => {
                        items.push(Self::build_loaded_item(
                            Some(path),
                            file_name,
                            None,
                            summary,
                        ));
                    }
                    Err(err) => {
                        errors.push(format!("{file_name}: {err}"));
                    }
                }
                done += 1;
                let _ = tx.send(WorkerEvent::Progress(ProgressUpdate {
                    current: done.min(total),
                    total,
                }));
                repaint_ctx.request_repaint();
            }

            for (path, file_name, bytes) in inputs {
                match inspect_pdf(&bytes) {
                    Ok(summary) => {
                        items.push(Self::build_loaded_item(
                            path,
                            file_name,
                            Some(Arc::new(bytes)),
                            summary,
                        ));
                    }
                    Err(err) => {
                        errors.push(format!("{file_name}: {err}"));
                    }
                }
                done += 1;
                let _ = tx.send(WorkerEvent::Progress(ProgressUpdate {
                    current: done.min(total),
                    total,
                }));
                repaint_ctx.request_repaint();
            }

            let _ = tx.send(WorkerEvent::LoadFinished { items, errors });
            repaint_ctx.request_repaint();
        }) {
            self.worker_rx = None;
            self.progress = None;
            self.error_message = Some("Failed to start PDF loader worker thread.".to_string());
        }
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

        let prev_active = self.active_index();
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

        if self.active_index() != prev_active {
            self.reset_preview_on_file_switch();
        }
    }

    fn select_all(&mut self) {
        if self.files.is_empty() {
            return;
        }
        let prev_active = self.active_index();
        self.selected_indices = (0..self.files.len()).collect();
        if self.selection_anchor.is_none_or(|a| a >= self.files.len()) {
            self.selection_anchor = Some(0);
        }
        if self.active_index() != prev_active {
            self.reset_preview_on_file_switch();
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

    fn affected_indices(&self) -> SmallVec<[usize; 8]> {
        if self.files.is_empty() {
            return SmallVec::new();
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
        if apply_all
            && (delta.target_mb_changed
                || (delta.target_mode_changed
                    && draft.target_mode == TargetMode::Megabytes
                    && self.shared_target_size_mb.is_none()))
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
                item.settings.options.color_depth = item
                    .settings
                    .options
                    .color_depth
                    .fallback_for(draft.options.format);
            }
            if delta.color_depth_changed {
                item.settings.options.color_depth = draft
                    .options
                    .color_depth
                    .fallback_for(item.settings.options.format);
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
            if delta.suffix_changed {
                item.settings.suffix = draft.suffix.clone();
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
        self.start_batch_compression(ctx);
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
                path: item.path.clone(),
                in_memory_bytes: item.in_memory_bytes.clone(),
                original_size_mb: item.original_size_mb,
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
        if !spawn_worker_thread("pdf_batch_compress_worker", move || {
            set_current_thread_below_normal();
            // Process strictly one PDF file at a time in sequential order so only one
            // lopdf::Document and its output buffer ever reside in RAM.
            for job in jobs {
                let current_idx = (job.index + 1).min(total_jobs);
                let _ = tx.send(WorkerEvent::Progress(ProgressUpdate {
                    current: current_idx,
                    total: total_jobs,
                }));
                repaint_ctx.request_repaint();

                let compressed = if let Some(path) = &job.path {
                    compress_pdf_path(path, &job.settings.options, |_| {})
                        .map_err(|e| e.to_string())
                } else if let Some(bytes) = &job.in_memory_bytes {
                    compress_pdf(bytes, &job.settings.options, |_| {}).map_err(|e| e.to_string())
                } else {
                    Err("No PDF source available for compression.".to_string())
                };

                let outcome = compressed.and_then(|(out_bytes, stats)| {
                    let write_res =
                        write_pdf_bytes_safely_indexed(&job.output_path, &out_bytes, job.index);
                    drop(out_bytes);
                    write_res?;
                    Ok(Box::new(CompressionResult::from_stats(
                        job.output_path,
                        stats,
                        &job.settings,
                        job.original_size_mb,
                    )))
                });

                let _ = tx.send(WorkerEvent::BatchItemFinished {
                    index: job.index,
                    outcome,
                });
                repaint_ctx.request_repaint();
            }

            let _ = tx.send(WorkerEvent::BatchFinished { folder });
            repaint_ctx.request_repaint();
        }) {
            self.worker_rx = None;
            self.progress = None;
            self.error_message = Some("Failed to start batch compression worker thread.".to_string());
        }
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
                        RichText::new("Select").size(14.5).strong(),
                    )
                    .min_size(egui::vec2(100.0, 38.0));

                    if ui.add_enabled(!busy, select_btn).clicked() {
                        let ctx = ui.ctx().clone();
                        self.open_file_dialog(&ctx);
                    }

                    ui.add_space(10.0);

                    const COMPRESS_BTN_WIDTH: f32 = 124.0;
                    let middle_w = (ui.available_width() - COMPRESS_BTN_WIDTH - 12.0).max(80.0);

                    if !self.files.is_empty() {
                        let cb_col_w = 114.0_f32.min(middle_w);
                        ui.allocate_ui_with_layout(
                            egui::vec2(middle_w, 42.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.allocate_ui_with_layout(
                                    egui::vec2(cb_col_w, 42.0),
                                    egui::Layout::top_down(egui::Align::Min),
                                    |ui| {
                                        ui.spacing_mut().item_spacing.y = 2.0;
                                        if self.files.len() > 1 {
                                            let prev = self.apply_for_all;
                                            let resp = ui.add_enabled(
                                                !busy,
                                                egui::Checkbox::new(
                                                    &mut self.apply_for_all,
                                                    "Apply for ALL",
                                                ),
                                            );
                                            if resp.changed() && self.apply_for_all && !prev {
                                                self.sync_all_from_active();
                                            } else if resp.changed() && !self.apply_for_all {
                                                self.shared_target_size_mb = None;
                                            }
                                        } else {
                                            let checkbox_row_height =
                                                ui.spacing().interact_size.y;
                                            ui.allocate_space(egui::vec2(
                                                0.0,
                                                checkbox_row_height,
                                            ));
                                        }
                                        ui.checkbox(&mut self.show_preview, "Show Preview");
                                    },
                                );
                            },
                        );
                    } else {
                        ui.allocate_ui_with_layout(
                            egui::vec2(middle_w, 38.0),
                            egui::Layout::top_down(egui::Align::Center),
                            |ui| {
                                ui.add_space(9.0);
                                ui.add(
                                    egui::Label::new(
                                        RichText::new("No PDF selected").strong().size(14.5),
                                    )
                                    .truncate(),
                                );
                            },
                        );
                    }

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        let can_compress = !self.files.is_empty() && !busy;
                        let action_btn = egui::Button::new(
                            RichText::new("Compress")
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

                if !self.files.is_empty() {
                    ui.add_space(10.0);
                    self.render_file_table(ui, busy);
                }
            });
    }

    fn render_file_table(&mut self, ui: &mut egui::Ui, busy: bool) {
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

        if self.hot_metrics.orig_mb.len() != self.files.len() {
            self.hot_metrics.sync_from(&self.files);
        }

        let max_target_mb = if !has_files {
            10.0
        } else if self.is_multi_file() && self.apply_for_all {
            self.hot_metrics.max_orig_mb()
        } else {
            active_idx
                .and_then(|idx| self.hot_metrics.orig_mb.get(idx).copied())
                .map(|mb| mb.max(1e-6))
                .unwrap_or(10.0)
        };

        Frame::new()
            .fill(card_fill)
            .stroke(Stroke::new(1.0, border_color))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.label(RichText::new("Compression").strong().size(15.0));
                ui.add_space(8.0);

                let mut delta = SettingsDelta::default();

                ui.add_enabled_ui(!busy && has_editable_target, |ui| {
                    const LABEL_COL_W: f32 = 158.0;
                    const VALUE_BOX_W: f32 = 64.0;
                    const GRID_GAP_X: f32 = 8.0;
                    const GRID_GAP_Y: f32 = 8.0;
                    let row_h = ui.spacing().interact_size.y.max(22.0);
                    let item_gap = ui.spacing().item_spacing.x;
                    let col1_w = (ui.available_width() - LABEL_COL_W - GRID_GAP_X).max(160.0);
                    let fixed_slider_width =
                        (col1_w - VALUE_BOX_W - item_gap - 18.0).clamp(80.0, 260.0);

                    let prev_filter = draft.options.resize_filter;
                    let prev_format = draft.options.format;
                    let prev_depth = draft.options.color_depth;
                    let prev_suffix = draft.suffix.clone();
                    let uses_quality = draft.options.format.uses_quality();
                    let prev_quality = draft.options.quality;
                    let prev_scale = draft.options.scale;

                    #[cfg(test)]
                    let mut captured_q_slider = None;
                    #[cfg(test)]
                    let mut captured_q_drag = None;
                    #[cfg(test)]
                    let mut captured_s_slider = None;
                    #[cfg(test)]
                    let mut captured_s_drag = None;

                    egui::Grid::new("compression_controls_grid")
                        .num_columns(2)
                        .spacing([GRID_GAP_X, GRID_GAP_Y])
                        .show(ui, |ui| {
                            // 1. Resize Filter
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Resize Filter:");
                                },
                            );
                            ui.horizontal(|ui| {
                                for filter in ResizeFilterChoice::ALL {
                                    let selected = draft.options.resize_filter == filter;
                                    if ui.selectable_label(selected, filter.label()).clicked() {
                                        draft.options.resize_filter = filter;
                                    }
                                }
                            });
                            ui.end_row();
                            if draft.options.resize_filter != prev_filter {
                                delta.filter_changed = true;
                            }

                            // 2. Image Format
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Image Format:");
                                },
                            );
                            egui::ComboBox::from_id_salt("image_format_selector")
                                .selected_text(draft.options.format.label())
                                .width(155.0)
                                .show_ui(ui, |ui| {
                                    for fmt in ImageFormatChoice::ALL {
                                        ui.selectable_value(
                                            &mut draft.options.format,
                                            fmt,
                                            fmt.label(),
                                        );
                                    }
                                });
                            ui.end_row();
                            if draft.options.format != prev_format {
                                delta.format_changed = true;
                                draft.options.color_depth = draft
                                    .options
                                    .color_depth
                                    .fallback_for(draft.options.format);
                            }

                            // 3. Color Depth
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Color Depth:");
                                },
                            );
                            let supported_depths = ColorDepth::supported_for(draft.options.format);
                            egui::ComboBox::from_id_salt("color_depth_selector")
                                .selected_text(draft.options.color_depth.label())
                                .width(155.0)
                                .show_ui(ui, |ui| {
                                    for &depth in supported_depths {
                                        ui.selectable_value(
                                            &mut draft.options.color_depth,
                                            depth,
                                            depth.label(),
                                        );
                                    }
                                });
                            ui.end_row();
                            if draft.options.color_depth != prev_depth {
                                delta.color_depth_changed = true;
                            }

                            // 4. Name Suffix
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Name Suffix:");
                                },
                            );
                            let suffix_box_fill = ui.visuals().widgets.inactive.bg_fill;
                            let suffix_box_stroke = ui.visuals().widgets.inactive.bg_stroke;
                            let suffix_frame_resp = Frame::new()
                                .fill(suffix_box_fill)
                                .stroke(suffix_box_stroke)
                                .corner_radius(CornerRadius::same(4))
                                .inner_margin(Margin::symmetric(6, 2))
                                .show(ui, |ui| {
                                    ui.add_sized(
                                        [143.0, (row_h - 6.0).max(14.0)],
                                        egui::TextEdit::singleline(&mut draft.suffix)
                                            .frame(Frame::NONE)
                                            .margin(Margin::ZERO)
                                            .background_color(Color32::TRANSPARENT)
                                            .hint_text("(original filename)"),
                                    )
                                });
                            if suffix_frame_resp.inner.changed() && draft.suffix != prev_suffix {
                                delta.suffix_changed = true;
                            }
                            ui.end_row();

                            // 5. Quality Factor
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Quality Factor:");
                                },
                            );
                            ui.horizontal(|ui| {
                                ui.spacing_mut().slider_width = fixed_slider_width;
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
                                            [VALUE_BOX_W, row_h],
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

                            // 6. Resolution Scale
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.label("Resolution Scale:");
                                },
                            );
                            ui.horizontal(|ui| {
                                ui.spacing_mut().slider_width = fixed_slider_width;
                                let scale_slider_resp = ui.add(
                                    egui::Slider::new(&mut draft.options.scale, 0.10..=1.00)
                                        .step_by(0.01)
                                        .show_value(false)
                                        .clamping(egui::SliderClamping::Edits),
                                );
                                let scale_drag_resp = ui.add_sized(
                                    [VALUE_BOX_W, row_h],
                                    egui::DragValue::new(&mut draft.options.scale)
                                        .range(0.10..=1.00)
                                        .speed(0.01)
                                        .fixed_decimals(2),
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

                            // 7. Target Size / Estimated Size
                            let prev_mode = draft.target_mode;
                            ui.allocate_ui_with_layout(
                                egui::vec2(LABEL_COL_W, row_h),
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.set_min_width(LABEL_COL_W);
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
                                },
                            );

                            match draft.target_mode {
                                TargetMode::Off => {
                                    ui.allocate_ui_with_layout(
                                        egui::vec2(col1_w, row_h),
                                        egui::Layout::left_to_right(egui::Align::Center),
                                        |ui| {
                                            self.render_estimated_size_inline(
                                                ui,
                                                active_idx,
                                                &draft.options,
                                            );
                                        },
                                    );
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
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().slider_width = fixed_slider_width;
                                            let slider_resp = ui.add(
                                                egui::Slider::new(
                                                    &mut draft.target_size_mb,
                                                    0.0..=max_target_mb,
                                                )
                                                .show_value(false)
                                                .clamping(egui::SliderClamping::Always),
                                            );
                                            let drag_resp = ui.add_sized(
                                                [VALUE_BOX_W, row_h],
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
                                                && (draft.target_size_mb - prev_target).abs()
                                                    >= 1e-5
                                            {
                                                delta.target_mb_changed = true;
                                            }
                                        });
                                    });
                                }
                                TargetMode::Percent => {
                                    draft.target_size_pct =
                                        draft.target_size_pct.clamp(1.0, 100.0);
                                    let prev_pct = draft.target_size_pct;

                                    ui.add_enabled_ui(has_files, |ui| {
                                        ui.horizontal(|ui| {
                                            ui.spacing_mut().slider_width = fixed_slider_width;
                                            let slider_resp = ui.add(
                                                egui::Slider::new(
                                                    &mut draft.target_size_pct,
                                                    1.0..=100.0,
                                                )
                                                .show_value(false)
                                                .step_by(1.0)
                                                .clamping(egui::SliderClamping::Always),
                                            );
                                            let drag_resp = ui.add_sized(
                                                [VALUE_BOX_W, row_h],
                                                egui::DragValue::new(&mut draft.target_size_pct)
                                                    .range(1.0..=100.0)
                                                    .speed(0.5)
                                                    .fixed_decimals(0)
                                                    .suffix("%"),
                                            );
                                            if (slider_resp.changed() || drag_resp.changed())
                                                && (draft.target_size_pct - prev_pct).abs()
                                                    >= 1e-4
                                            {
                                                delta.target_pct_changed = true;
                                            }
                                        });
                                    });
                                }
                            }
                            ui.end_row();
                        });

                    #[cfg(test)]
                    {
                        self.last_quality_slider_rect = captured_q_slider;
                        self.last_quality_drag_rect = captured_q_drag;
                        self.last_scale_slider_rect = captured_s_slider;
                        self.last_scale_drag_rect = captured_s_drag;
                    }
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

    fn render_summary_card(&mut self, ui: &mut egui::Ui, panel_bottom_y: f32) -> f32 {
        const SUMMARY_FOOTER_HEIGHT: f32 = 22.0;
        const SUMMARY_FOOTER_TOP_GAP: f32 = 6.0;
        const SUMMARY_BOTTOM_RESERVE: f32 =
            SUMMARY_FOOTER_TOP_GAP + SUMMARY_FOOTER_HEIGHT + 14.0 + 2.0;

        let dark = ui.visuals().dark_mode;
        let busy = self.is_busy();
        let mut clear_clicked = false;
        let mut extra_summary_h = 0.0_f32;
        Frame::new()
            .fill(neutral_card_fill(dark))
            .stroke(Stroke::new(1.0, neutral_card_border(dark)))
            .corner_radius(CornerRadius::same(10))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                ui.horizontal(|ui| {
                    ui.label(RichText::new("Compression Summary").strong().size(15.0));
                    if busy {
                        self.render_summary_header_progress(ui);
                    } else if !self.files.is_empty() {
                        ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                            if ui
                                .add(
                                    egui::Button::new(RichText::new("Clear").size(11.5))
                                        .min_size(egui::vec2(44.0, 18.0)),
                                )
                                .on_hover_text("Clear loaded PDFs and free preview memory")
                                .clicked()
                            {
                                clear_clicked = true;
                            }
                        });
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

                let file_count = self.files.len();
                let monitor_height_pts = ui
                    .ctx()
                    .input(|i| i.viewport().monitor_size.map(|m| m.y))
                    .filter(|h| h.is_finite() && *h > 0.0)
                    .unwrap_or_else(|| 1080.0 / ui.ctx().pixels_per_point().max(1.0));
                let display_max_rows = if monitor_height_pts <= 1100.0 {
                    9
                } else {
                    SUMMARY_TABLE_MAX_VISIBLE_ROWS
                };
                let row_stride = TABLE_ROW_HEIGHT + TABLE_ROW_GAP;
                let avail_rows_height =
                    (panel_bottom_y - ui.cursor().top() - SUMMARY_BOTTOM_RESERVE).max(0.0);
                let fitting_rows = (((avail_rows_height + TABLE_ROW_GAP) / row_stride)
                    .floor()
                    .max(1.0)) as usize;

                let (scroll_h, auto_shrink_y) = if file_count < SUMMARY_TABLE_MIN_VISIBLE_ROWS {
                    (summary_rows_height(file_count), true)
                } else {
                    let max_summary_rows =
                        fitting_rows.clamp(SUMMARY_TABLE_MIN_VISIBLE_ROWS, display_max_rows);
                    let visible_summary_rows = file_count.min(max_summary_rows);
                    let min_scroll_h = summary_rows_height(SUMMARY_TABLE_MIN_VISIBLE_ROWS);
                    let target_h = summary_rows_height(visible_summary_rows);
                    extra_summary_h = (target_h - min_scroll_h).max(0.0);
                    (target_h, false)
                };
                let guide_stroke = Stroke::new(1.0, table_row_guide_color(dark));

                let mut scroll_area = egui::ScrollArea::vertical()
                    .id_salt("batch_summary_scroll")
                    .max_height(scroll_h)
                    .auto_shrink([false, auto_shrink_y]);
                if !auto_shrink_y {
                    scroll_area = scroll_area.min_scrolled_height(scroll_h);
                }
                scroll_area.show(ui, |ui| {
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

                // Always reserve the fixed footer band for "Saved to: ..." so completing
                // compression never shifts the card height or overflows the bottom window border.
                ui.add_space(SUMMARY_FOOTER_TOP_GAP);
                let saved_text: Option<String> = if self.files.len() == 1 {
                    self.files
                        .first()
                        .and_then(|f| f.result.as_ref())
                        .map(|res| format!("Saved to: {}", res.saved_path.display()))
                } else {
                    self.batch_saved_folder
                        .as_ref()
                        .map(|folder| format!("Saved to: {}", folder.display()))
                };
                ui.allocate_ui_with_layout(
                    egui::vec2(ui.available_width(), SUMMARY_FOOTER_HEIGHT),
                    egui::Layout::left_to_right(egui::Align::Center),
                    |ui| {
                        if let Some(text) = saved_text {
                            ui.add(
                                egui::Label::new(
                                    RichText::new(text)
                                        .size(11.5)
                                        .color(ui.visuals().weak_text_color()),
                                )
                                .truncate(),
                            );
                        }
                    },
                );
            });

        if clear_clicked {
            self.clear_files();
        }
        extra_summary_h
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

    fn sync_preview_selection_and_poll(&mut self, ctx: &egui::Context) {
        let Some(active_idx) = (if self.show_preview {
            self.active_index()
        } else {
            None
        }) else {
            if self.preview_display.is_some()
                || self.preview_pending_key.is_some()
                || self.preview_inflight_key.is_some()
                || self.preview_rx.is_some()
            {
                self.clear_preview_textures();
            }
            return;
        };

        let total_pages = self.files[active_idx].summary.page_count.max(1);
        self.preview_page = self.preview_page.clamp(1, total_pages);
        let desired_key =
            PreviewKey::from_item(active_idx, &self.files[active_idx], self.preview_page);

        // Immediately drop old page/file textures when switching pages or active files.
        if self
            .preview_display
            .as_ref()
            .is_some_and(|d| !d.key.same_source_page(&desired_key))
        {
            self.preview_display = None;
            self.preview_show_original_held = false;
            self.preview_scroll_offset = egui::Vec2::ZERO;
        }

        // Poll completed background preview jobs and upload single-page RGBA textures.
        let current_gen = self.preview_generation.load(Ordering::Relaxed);
        let mut completed: Vec<PreviewWorkerEvent> = Vec::new();
        if let Some(rx) = &self.preview_rx {
            while let Ok(ev) = rx.try_recv() {
                completed.push(ev);
            }
        }
        for ev in completed {
            if self.preview_inflight_key.as_ref() == Some(&ev.key) {
                self.preview_inflight_key = None;
            }
            if ev.generation != current_gen || !ev.key.same_source_page(&desired_key) {
                continue;
            }
            // Release previous TextureHandles before allocating the next page's textures,
            // while preserving the existing Original texture handle for same-page slider ticks.
            let prev_same_page_orig_tex = self.preview_display.take().and_then(|d| {
                if d.key.same_source_page(&ev.key) {
                    d.orig_texture
                } else {
                    None
                }
            });
            const MAX_SAFE_GL_TEX_DIM: u32 = 4096;
            let mut missing_reused_orig = false;
            let display = match ev.outcome {
                Ok(None) => PreviewDisplay {
                    key: ev.key,
                    orig_texture: None,
                    orig_width: 0,
                    orig_height: 0,
                    comp_texture: None,
                    comp_width: 0,
                    comp_height: 0,
                    kept_original: false,
                    synthetic_empty: true,
                    error: None,
                },
                Ok(Some(page_data)) => {
                    let PagePreviewData {
                        orig_width,
                        orig_height,
                        orig_stream_bytes: _,
                        orig_rgba,
                        comp_width,
                        comp_height,
                        comp_stream_bytes: _,
                        comp_rgba,
                        kept_original,
                    } = *page_data;
                    let expected_orig_len =
                        (orig_width as usize).saturating_mul(orig_height as usize).saturating_mul(4);
                    let orig_rgba_was_empty = orig_rgba.is_empty();
                    let orig_texture = if orig_width > 0
                        && orig_height > 0
                        && orig_width <= MAX_SAFE_GL_TEX_DIM
                        && orig_height <= MAX_SAFE_GL_TEX_DIM
                        && orig_rgba.len() == expected_orig_len
                    {
                        let orig_img = egui::ColorImage::from_rgba_unmultiplied(
                            [orig_width as usize, orig_height as usize],
                            &orig_rgba,
                        );
                        drop(orig_rgba);
                        Some(ctx.load_texture(
                            "pdf_preview_original",
                            orig_img,
                            egui::TextureOptions::LINEAR,
                        ))
                    } else if orig_rgba_was_empty {
                        prev_same_page_orig_tex
                    } else {
                        None
                    };
                    if orig_rgba_was_empty && orig_texture.is_none() {
                        missing_reused_orig = true;
                    }
                    let comp_texture = if let Some(comp_rgba) = comp_rgba {
                        let expected_comp_len = (comp_width as usize)
                            .saturating_mul(comp_height as usize)
                            .saturating_mul(4);
                        if comp_width > 0
                            && comp_height > 0
                            && comp_width <= MAX_SAFE_GL_TEX_DIM
                            && comp_height <= MAX_SAFE_GL_TEX_DIM
                            && comp_rgba.len() == expected_comp_len
                        {
                            let comp_img = egui::ColorImage::from_rgba_unmultiplied(
                                [comp_width as usize, comp_height as usize],
                                &comp_rgba,
                            );
                            drop(comp_rgba);
                            Some(ctx.load_texture(
                                "pdf_preview_compressed",
                                comp_img,
                                egui::TextureOptions::LINEAR,
                            ))
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    PreviewDisplay {
                        key: ev.key,
                        orig_texture,
                        orig_width,
                        orig_height,
                        comp_texture,
                        comp_width,
                        comp_height,
                        kept_original,
                        synthetic_empty: false,
                        error: None,
                    }
                }
                Err(err) => PreviewDisplay {
                    key: ev.key,
                    orig_texture: None,
                    orig_width: 0,
                    orig_height: 0,
                    comp_texture: None,
                    comp_width: 0,
                    comp_height: 0,
                    kept_original: false,
                    synthetic_empty: false,
                    error: Some(err),
                },
            };
            if missing_reused_orig {
                if let Some(tx) = &self.preview_cmd_tx {
                    let _ = tx.send(PreviewCommand::ClearCache);
                }
            } else {
                self.preview_display = Some(Box::new(display));
            }
        }

        // Synthetic test items have neither a file path nor raw bytes; avoid spawning error workers.
        let item = &self.files[active_idx];
        if item.path.is_none() && item.in_memory_bytes.is_none() {
            if self.preview_display.as_ref().is_none_or(|d| d.key != desired_key) {
                self.preview_display = Some(Box::new(PreviewDisplay {
                    key: desired_key,
                    orig_texture: None,
                    orig_width: 0,
                    orig_height: 0,
                    comp_texture: None,
                    comp_width: 0,
                    comp_height: 0,
                    kept_original: false,
                    synthetic_empty: true,
                    error: None,
                }));
            }
            return;
        }

        let up_to_date = self
            .preview_display
            .as_ref()
            .is_some_and(|d| d.key == desired_key)
            || self.preview_inflight_key.as_ref() == Some(&desired_key);
        if up_to_date {
            self.preview_pending_key = None;
            self.preview_pending_since = None;
            if self.preview_inflight_key.is_some() {
                ctx.request_repaint_after(Duration::from_millis(30));
            }
            return;
        }

        // Pause launching new preview renders while PDF loading or compression is actively running
        // so background workers never compete for CPU/RAM on low-end hardware.
        if self.is_busy() {
            return;
        }

        // Debounce ~120 ms for slider/setting changes; dispatch immediately on page/file switch.
        let same_page_displayed = self
            .preview_display
            .as_ref()
            .is_some_and(|d| d.key.same_source_page(&desired_key))
            || self
                .preview_inflight_key
                .as_ref()
                .is_some_and(|k| k.same_source_page(&desired_key));

        if self.preview_pending_key.as_ref() != Some(&desired_key) {
            self.preview_pending_key = Some(desired_key.clone());
            self.preview_pending_since = Some(Instant::now());
        }

        let elapsed = self
            .preview_pending_since
            .map_or(Duration::ZERO, |t| t.elapsed());
        let debounce = if same_page_displayed {
            Duration::from_millis(PREVIEW_DEBOUNCE_MS)
        } else {
            Duration::ZERO
        };

        if elapsed >= debounce {
            self.preview_pending_key = None;
            self.preview_pending_since = None;
            self.spawn_preview_worker(active_idx, desired_key, ctx);
        } else {
            let remaining = debounce.saturating_sub(elapsed).max(Duration::from_millis(16));
            ctx.request_repaint_after(remaining);
        }
    }

    fn ensure_preview_worker(&mut self) {
        if self.preview_cmd_tx.is_some() {
            return;
        }
        let (cmd_tx, cmd_rx) = mpsc::channel::<PreviewCommand>();
        let (ev_tx, ev_rx) = mpsc::channel::<PreviewWorkerEvent>();
        let gen_counter = Arc::clone(&self.preview_generation);
        self.preview_cmd_tx = Some(cmd_tx);
        self.preview_rx = Some(ev_rx);

        let spawn_worker = move || {
            set_current_thread_below_normal();
            let mut cache = PreviewSessionCache::default();
            while let Ok(mut cmd) = cmd_rx.recv() {
                // Coalesce bursty slider ticks / page changes to only the newest pending command.
                while let Ok(newer) = cmd_rx.try_recv() {
                    cmd = newer;
                }
                match cmd {
                    PreviewCommand::ClearCache => {
                        cache.clear();
                    }
                    PreviewCommand::Render(req) => {
                        let PreviewRenderRequest {
                            generation,
                            key,
                            path,
                            in_memory_bytes,
                            options,
                            reply_ctx,
                        } = *req;
                        let is_cancelled = || gen_counter.load(Ordering::Relaxed) != generation;
                        if is_cancelled() {
                            continue;
                        }
                        let page_number = key.page_number;
                        let source_id = key.source_id;
                        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            if let Some(ref path) = path {
                                render_page_preview_cached(
                                    &mut cache,
                                    source_id,
                                    PreviewSource::Path(path.as_path()),
                                    page_number,
                                    &options,
                                    &is_cancelled,
                                )
                                .map(|data| Some(Box::new(data)))
                                .map_err(|e| e.to_string())
                            } else if let Some(bytes) = in_memory_bytes {
                                render_page_preview_cached(
                                    &mut cache,
                                    source_id,
                                    PreviewSource::SharedBytes(bytes),
                                    page_number,
                                    &options,
                                    &is_cancelled,
                                )
                                .map(|data| Some(Box::new(data)))
                                .map_err(|e| e.to_string())
                            } else {
                                Ok(None)
                            }
                        }))
                        .unwrap_or_else(|_| {
                            cache.clear();
                            Err("Preview unavailable for this page".to_string())
                        });

                        let _ = ev_tx.send(PreviewWorkerEvent {
                            generation,
                            key,
                            outcome,
                        });
                        reply_ctx.request_repaint();
                    }
                }
            }
        };
        if !spawn_worker_thread("pdf_preview_worker", spawn_worker) {
            self.preview_cmd_tx = None;
            self.preview_rx = None;
        }
    }

    fn spawn_preview_worker(
        &mut self,
        active_idx: usize,
        key: PreviewKey,
        ctx: &egui::Context,
    ) {
        let Some((path, in_memory_bytes, options)) = self.files.get(active_idx).map(|item| {
            (
                item.path.clone(),
                item.in_memory_bytes.clone(),
                item.settings.options.clone(),
            )
        }) else {
            return;
        };

        self.ensure_preview_worker();
        let gen = self.preview_generation.fetch_add(1, Ordering::Relaxed) + 1;
        self.preview_inflight_key = Some(key.clone());

        if let Some(tx) = &self.preview_cmd_tx {
            let _ = tx.send(PreviewCommand::Render(Box::new(PreviewRenderRequest {
                generation: gen,
                key,
                path,
                in_memory_bytes,
                options,
                reply_ctx: ctx.clone(),
            })));
        }
    }

    fn active_preview_view_mode(&self) -> PreviewViewMode {
        if self.preview_fixed_zoom {
            PreviewViewMode::FixedZoomPct(self.preview_zoom_pct.clamp(10, 500))
        } else {
            PreviewViewMode::Framing(self.preview_framing)
        }
    }

    fn commit_preview_zoom_edit(&mut self) {
        let cleaned = self.preview_zoom_edit_buf.trim().trim_end_matches('%').trim();
        if let Ok(val) = cleaned.parse::<f32>() {
            if val.is_finite() {
                self.preview_zoom_exact = val.clamp(10.0, 500.0);
                self.preview_zoom_pct = self.preview_zoom_exact.round().clamp(10.0, 500.0) as u16;
                self.preview_fixed_zoom = true;
            }
        }
        self.preview_zoom_editing = false;
        self.preview_zoom_just_started = false;
    }

    fn commit_preview_page_edit(&mut self, total_pages: usize) {
        let trimmed = self.preview_page_edit_buf.trim();
        if let Ok(val) = trimmed.parse::<usize>() {
            self.preview_page = val.clamp(1, total_pages.max(1));
        }
        self.preview_page_edit_buf = self.preview_page.to_string();
    }

    fn render_preview_panel(&mut self, ui: &mut egui::Ui) {
        let Some(active_idx) = self.active_index() else {
            return;
        };
        let total_pages = self.files[active_idx].summary.page_count.max(1);
        self.preview_page = self.preview_page.clamp(1, total_pages);
        let busy_preview =
            self.preview_inflight_key.is_some() || self.preview_pending_key.is_some();
        let dark = ui.visuals().dark_mode;

        // Header bar: 3 responsive zones — left title, dynamically centered page selector, right [100%] [Fit] [Original].
        const TOOLBAR_H: f32 = 24.0;
        const GAP_X: f32 = 6.0;
        const FRAMING_W: f32 = 48.0;
        const ZOOM_W: f32 = 58.0;
        const ORIG_W: f32 = 62.0;

        let avail_w = ui.available_width().max(120.0);
        let (header_rect, _) =
            ui.allocate_exact_size(egui::vec2(avail_w, TOOLBAR_H), egui::Sense::hover());

        // 1. Left Zone: "Page Preview" + optional spinner.
        let left_resp = ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(header_rect)
                .layout(egui::Layout::left_to_right(egui::Align::Center)),
            |ui| {
                ui.spacing_mut().item_spacing.x = GAP_X;
                ui.label(RichText::new("Page Preview").strong().size(15.0));
                if busy_preview {
                    ui.add(egui::Spinner::new().size(13.0));
                }
            },
        );
        let left_end_x = left_resp.response.rect.max.x;

        // 3. Right Zone geometry: right-to-left anchored at header_rect.max.x so zoom textbox expands left without moving Fit/Original.
        let right_reserve_w = ZOOM_W + FRAMING_W + ORIG_W + 2.0 * GAP_X;
        let right_start_x = (header_rect.max.x - right_reserve_w).max(header_rect.min.x);
        let right_rect = egui::Rect::from_min_max(
            egui::pos2(left_end_x + GAP_X, header_rect.min.y),
            egui::pos2(header_rect.max.x, header_rect.min.y + TOOLBAR_H),
        );

        // 2. Center Zone geometry: < [ 1 / X ] > centered in the preview panel, clamped away from left/right zones.
        let total_str = total_pages.to_string();
        let digits_w = (total_str.len() as f32 * 7.5).max(12.0);
        let pill_w = 18.0 + digits_w * 2.0 + 16.0;
        let center_w = TOOLBAR_H + pill_w + TOOLBAR_H + 2.0 * GAP_X;
        let ideal_center_x = header_rect.center().x - center_w * 0.5;
        let min_center_x = left_end_x + GAP_X;
        let max_center_x = (right_start_x - GAP_X - center_w).max(min_center_x);
        let center_x = ideal_center_x.clamp(min_center_x, max_center_x);
        let center_avail_w = (right_start_x - GAP_X - center_x).max(0.0).min(center_w);
        if center_avail_w >= 40.0 {
            let center_rect = egui::Rect::from_min_size(
                egui::pos2(center_x, header_rect.min.y),
                egui::vec2(center_avail_w, TOOLBAR_H),
            );
            ui.scope_builder(
                egui::UiBuilder::new()
                    .max_rect(center_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
                |ui| {
                    ui.spacing_mut().item_spacing.x = GAP_X;
                    ui.spacing_mut().interact_size.y = TOOLBAR_H;

                    if ui
                        .add_enabled(
                            self.preview_page > 1 && total_pages > 1,
                            egui::Button::new("<")
                                .min_size(egui::vec2(TOOLBAR_H, TOOLBAR_H))
                                .corner_radius(CornerRadius::same(8)),
                        )
                        .on_hover_text("Previous page")
                        .clicked()
                    {
                        self.preview_page = self.preview_page.saturating_sub(1).max(1);
                        self.preview_page_edit_buf = self.preview_page.to_string();
                    }

                    let pill_fill = if dark {
                        Color32::from_rgb(38, 43, 52)
                    } else {
                        Color32::from_rgb(236, 240, 246)
                    };
                    Frame::new()
                        .fill(pill_fill)
                        .stroke(Stroke::new(1.0, neutral_card_border(dark)))
                        .corner_radius(CornerRadius::same(12))
                        .inner_margin(Margin::symmetric(9, 0))
                        .show(ui, |ui| {
                            ui.set_height(TOOLBAR_H - 2.0);
                            ui.with_layout(
                                egui::Layout::left_to_right(egui::Align::Center),
                                |ui| {
                                    ui.spacing_mut().item_spacing.x = 3.0;
                                    let page_edit_id =
                                        ui.make_persistent_id("preview_page_jump_edit");
                                    if !ui.ctx().memory(|m| m.has_focus(page_edit_id)) {
                                        let current_text = self.preview_page.to_string();
                                        if self.preview_page_edit_buf != current_text {
                                            self.preview_page_edit_buf = current_text;
                                        }
                                    }
                                    let edit_resp = ui.add(
                                        egui::TextEdit::singleline(
                                            &mut self.preview_page_edit_buf,
                                        )
                                        .id(page_edit_id)
                                        .frame(Frame::NONE)
                                        .margin(Margin::ZERO)
                                        .background_color(Color32::TRANSPARENT)
                                        .font(egui::FontId::proportional(12.0))
                                        .align(egui::Align2::CENTER_CENTER)
                                        .desired_width(digits_w)
                                        .interactive(total_pages > 1),
                                    );
                                    if edit_resp.has_focus()
                                        && ui.input(|i| i.key_pressed(egui::Key::Escape))
                                    {
                                        self.preview_page_edit_buf =
                                            self.preview_page.to_string();
                                        edit_resp.surrender_focus();
                                    } else if edit_resp.lost_focus()
                                        || (edit_resp.has_focus()
                                            && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                                    {
                                        self.commit_preview_page_edit(total_pages);
                                    }
                                    ui.label(
                                        RichText::new("/")
                                            .size(12.0)
                                            .color(ui.visuals().weak_text_color()),
                                    );
                                    ui.label(
                                        RichText::new(&total_str)
                                            .size(12.0)
                                            .color(ui.visuals().weak_text_color()),
                                    );
                                },
                            );
                        });

                    if ui
                        .add_enabled(
                            self.preview_page < total_pages,
                            egui::Button::new(">")
                                .min_size(egui::vec2(TOOLBAR_H, TOOLBAR_H))
                                .corner_radius(CornerRadius::same(8)),
                        )
                        .on_hover_text("Next page")
                        .clicked()
                    {
                        self.preview_page = (self.preview_page + 1).min(total_pages);
                        self.preview_page_edit_buf = self.preview_page.to_string();
                    }
                },
            );
        }

        // Render Right Zone anchored on right edge (right-to-left): [Original] <- [Fit] <- [100%/edit]
        ui.scope_builder(
            egui::UiBuilder::new()
                .max_rect(right_rect)
                .layout(egui::Layout::right_to_left(egui::Align::Center)),
            |ui| {
                ui.spacing_mut().item_spacing.x = GAP_X;
                ui.spacing_mut().interact_size.y = TOOLBAR_H;

                let has_orig = self
                    .preview_display
                    .as_ref()
                    .is_some_and(|d| d.orig_texture.is_some());
                let prev_held = self.preview_show_original_held && has_orig;
                let accent_fill = ui.visuals().selection.bg_fill;
                let accent_stroke = ui.visuals().selection.stroke;
                let mut orig_btn = egui::Button::new(
                    RichText::new("Original").color(if prev_held {
                        accent_stroke.color
                    } else {
                        ui.visuals().text_color()
                    }),
                )
                .min_size(egui::vec2(ORIG_W, TOOLBAR_H))
                .sense(egui::Sense::click_and_drag());
                if prev_held {
                    orig_btn = orig_btn.fill(accent_fill).stroke(accent_stroke);
                }
                let orig_resp = ui
                    .add_enabled(has_orig, orig_btn)
                    .on_hover_text("Hold to preview the uncompressed original page");
                let held_now = has_orig
                    && orig_resp.is_pointer_button_down_on()
                    && ui.input(|i| i.pointer.primary_down());
                if held_now != self.preview_show_original_held {
                    self.preview_show_original_held = held_now;
                    ui.ctx().request_repaint();
                }

                if ui
                    .add_sized(
                        [FRAMING_W, TOOLBAR_H],
                        egui::Button::selectable(
                            !self.preview_fixed_zoom,
                            self.preview_framing.label(),
                        ),
                    )
                    .on_hover_text(self.preview_framing.tooltip())
                    .clicked()
                {
                    if self.preview_fixed_zoom {
                        self.preview_fixed_zoom = false;
                        self.preview_zoom_editing = false;
                    } else {
                        self.preview_framing = self.preview_framing.next();
                    }
                }

                const ZOOM_H: f32 = 22.0;
                let (pill_rect, pill_resp) =
                    ui.allocate_exact_size(egui::vec2(ZOOM_W, ZOOM_H), egui::Sense::click());

                let (pill_fill, pill_stroke) = if self.preview_zoom_editing {
                    (
                        if dark {
                            Color32::from_rgb(38, 43, 52)
                        } else {
                            Color32::from_rgb(236, 240, 246)
                        },
                        Stroke::new(1.0, ui.visuals().selection.stroke.color),
                    )
                } else if self.preview_fixed_zoom {
                    (
                        ui.visuals().selection.bg_fill,
                        ui.visuals().selection.stroke,
                    )
                } else if pill_resp.hovered() {
                    (
                        ui.visuals().widgets.hovered.bg_fill,
                        ui.visuals().widgets.hovered.bg_stroke,
                    )
                } else {
                    (
                        if dark {
                            Color32::from_rgb(38, 43, 52)
                        } else {
                            Color32::from_rgb(236, 240, 246)
                        },
                        Stroke::new(1.0, neutral_card_border(dark)),
                    )
                };

                ui.painter().rect(
                    pill_rect,
                    CornerRadius::same(4),
                    pill_fill,
                    pill_stroke,
                    egui::StrokeKind::Inside,
                );

                if self.preview_zoom_editing {
                    let inner_rect = pill_rect.shrink2(egui::vec2(6.0, 2.0));
                    ui.scope_builder(
                        egui::UiBuilder::new()
                            .max_rect(inner_rect)
                            .layout(egui::Layout::left_to_right(egui::Align::Center)),
                        |ui| {
                            ui.set_clip_rect(inner_rect);
                            ui.spacing_mut().item_spacing.x = 1.0;
                            ui.spacing_mut().interact_size.y = inner_rect.height();
                            let edit_resp = ui.add(
                                egui::TextEdit::singleline(&mut self.preview_zoom_edit_buf)
                                    .frame(Frame::NONE)
                                    .margin(Margin::ZERO)
                                    .background_color(Color32::TRANSPARENT)
                                    .font(egui::FontId::proportional(12.0))
                                    .align(egui::Align2::RIGHT_CENTER)
                                    .desired_width(32.0)
                                    .hint_text("100"),
                            );
                            ui.label(
                                RichText::new("%")
                                    .size(12.0)
                                    .color(ui.visuals().weak_text_color()),
                            );
                            if self.preview_zoom_just_started {
                                edit_resp.request_focus();
                                self.preview_zoom_just_started = false;
                            } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                                self.preview_zoom_editing = false;
                            } else if edit_resp.lost_focus()
                                || (edit_resp.has_focus()
                                    && ui.input(|i| i.key_pressed(egui::Key::Enter)))
                            {
                                self.commit_preview_zoom_edit();
                            }
                        },
                    );
                } else {
                    let zoom_label = if self.preview_fixed_zoom {
                        format!("{}%", self.preview_zoom_pct)
                    } else {
                        "100%".to_string()
                    };
                    let zoom_hint = if self.preview_fixed_zoom {
                        "Fixed zoom active — click again to type a custom percentage (10%–500%)"
                    } else {
                        "Switch to fixed 100% (1.0×) zoom"
                    };
                    let text_color = if self.preview_fixed_zoom {
                        ui.visuals().selection.stroke.color
                    } else if pill_resp.hovered() {
                        ui.visuals().widgets.hovered.fg_stroke.color
                    } else {
                        ui.visuals().text_color()
                    };
                    ui.painter().text(
                        pill_rect.center(),
                        egui::Align2::CENTER_CENTER,
                        zoom_label,
                        egui::FontId::proportional(12.0),
                        text_color,
                    );
                    if pill_resp.on_hover_text(zoom_hint).clicked() {
                        if !self.preview_fixed_zoom {
                            self.preview_fixed_zoom = true;
                            self.preview_zoom_pct = 100;
                            self.preview_zoom_exact = 100.0;
                        } else {
                            self.preview_zoom_editing = true;
                            self.preview_zoom_just_started = true;
                            self.preview_zoom_edit_buf = self.preview_zoom_pct.to_string();
                        }
                    }
                }
            },
        );

        ui.add_space(8.0);

        let card_fill = neutral_card_fill(dark);
        let border_color = neutral_card_border(dark);
        let weak_color = ui.visuals().weak_text_color();

        let Some(display) = &self.preview_display else {
            Self::render_preview_placeholder(
                ui,
                card_fill,
                border_color,
                weak_color,
                if busy_preview {
                    "Rendering page preview..."
                } else {
                    "Preparing preview..."
                },
                busy_preview,
            );
            return;
        };

        if let Some(_err) = &display.error {
            Self::render_preview_placeholder(
                ui,
                card_fill,
                border_color,
                Color32::from_rgb(200, 75, 75),
                "Preview unavailable for this page",
                false,
            );
            return;
        }

        if display.synthetic_empty {
            Self::render_preview_placeholder(
                ui,
                card_fill,
                border_color,
                weak_color,
                "Preview unavailable (no PDF data loaded)",
                false,
            );
            return;
        }

        let orig_tex = display
            .orig_texture
            .as_ref()
            .map(|t| (t.id(), display.orig_width, display.orig_height));
        let comp_tex = display
            .comp_texture
            .as_ref()
            .or({
                if display.kept_original
                    || (display.comp_width == display.orig_width
                        && display.comp_height == display.orig_height)
                {
                    display.orig_texture.as_ref()
                } else {
                    None
                }
            })
            .map(|t| (t.id(), display.comp_width, display.comp_height));

        let active_tex = if self.preview_show_original_held {
            orig_tex.or(comp_tex)
        } else {
            comp_tex.or(orig_tex)
        };

        let view_mode = self.active_preview_view_mode();
        let pane_height = ui.available_height().max(220.0);
        let pane_width = ui.available_width().max(120.0);

        let mut zoomed_pct = None;
        ui.allocate_ui_with_layout(
            egui::vec2(pane_width, pane_height),
            egui::Layout::top_down(egui::Align::Min),
            |ui| {
                zoomed_pct = render_preview_pane(
                    ui,
                    active_tex,
                    view_mode,
                    self.preview_zoom_exact,
                    &mut self.preview_scroll_offset,
                    card_fill,
                    border_color,
                );
            },
        );
        if let Some(new_exact_pct) = zoomed_pct {
            self.preview_fixed_zoom = true;
            self.preview_zoom_editing = false;
            self.preview_zoom_exact = new_exact_pct.clamp(10.0, 500.0);
            self.preview_zoom_pct = self.preview_zoom_exact.round().clamp(10.0, 500.0) as u16;
            ui.ctx().request_repaint();
        }
    }

    fn render_preview_placeholder(
        ui: &mut egui::Ui,
        card_fill: Color32,
        border_color: Color32,
        text_color: Color32,
        message: &str,
        show_spinner: bool,
    ) {
        let h = ui.available_height().max(220.0);
        Frame::new()
            .fill(card_fill)
            .stroke(Stroke::new(1.0, border_color))
            .corner_radius(CornerRadius::same(8))
            .inner_margin(Margin::same(14))
            .show(ui, |ui| {
                let inner_w = ui.available_width().max(40.0);
                let inner_h = (h - 30.0).max(160.0);
                ui.set_min_size(egui::vec2(inner_w, inner_h));
                ui.set_max_width(inner_w);
                ui.vertical_centered(|ui| {
                    ui.add_space(((inner_h - 28.0) * 0.5).max(0.0));
                    if show_spinner {
                        ui.add(egui::Spinner::new().size(15.0));
                        ui.add_space(4.0);
                    }
                    ui.add(
                        egui::Label::new(RichText::new(message).size(13.0).color(text_color))
                            .wrap(),
                    );
                });
            });
    }

    fn render_ui(&mut self, ui: &mut egui::Ui) {
        let ctx = ui.ctx().clone();
        self.poll_worker();
        self.handle_drag_and_drop(&ctx);
        self.sync_preview_selection_and_poll(&ctx);

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
        let show_preview = self.show_preview && self.active_index().is_some();
        let (main_width, default_preview_width) = dpi_scaled_widths(&ctx);

        let panel_frame = Frame::central_panel(ui.style()).inner_margin(Margin::same(10));
        let panel_v_margin =
            f32::from(panel_frame.inner_margin.top) + f32::from(panel_frame.inner_margin.bottom);

        let panel_resp = egui::Panel::left("main_panel")
            .exact_size(main_width)
            .resizable(false)
            .frame(panel_frame)
            .show(ui, |ui| {
                let panel_bottom_y = ui.max_rect().max.y;
                let inner_resp = ui.vertical(|ui| {
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

                    if !self.files.is_empty() {
                        ui.add_space(10.0);
                        self.render_summary_card(ui, panel_bottom_y)
                    } else {
                        0.0
                    }
                });

                let content_width = inner_resp.response.rect.width();
                let content_height = inner_resp.response.rect.height();
                let extra_summary_h = inner_resp.inner;
                (content_width, content_height, extra_summary_h)
            });

        if show_preview {
            egui::CentralPanel::default()
                .frame(
                    Frame::new()
                        .fill(ui.visuals().panel_fill)
                        .inner_margin(Margin::same(10)),
                )
                .show(ui, |ui| {
                    ui.set_width(ui.available_width());
                    self.render_preview_panel(ui);
                });
        } else {
            egui::CentralPanel::default()
                .frame(Frame::new().fill(ui.visuals().panel_fill))
                .show(ui, |_ui| {});
        }

        let (content_width, content_height, extra_summary_h) = panel_resp.inner;
        #[cfg(test)]
        {
            self.last_content_width = content_width;
        }
        #[cfg(not(test))]
        let _ = content_width;
        let min_content_height = (content_height - extra_summary_h).max(0.0);
        let max_fit_h = ctx
            .input(|i| i.viewport().monitor_size.map(|m| (m.y - 80.0).max(360.0)))
            .unwrap_or(960.0);
        let min_window_height = if min_content_height.is_finite() && min_content_height > 0.0 {
            (min_content_height + panel_v_margin + 2.0)
                .ceil()
                .min(max_fit_h)
        } else {
            MIN_WINDOW_HEIGHT
        };

        let preview_toggled = self.preview_panel_open != show_preview;
        let lock_key = (
            show_preview,
            main_width.round() as u32,
            min_window_height.round() as u32,
        );
        let lock_changed = self.last_viewport_lock_state != Some(lock_key);
        if lock_changed {
            self.last_viewport_lock_state = Some(lock_key);
            if show_preview {
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                    main_width + PREVIEW_MIN_WIDTH,
                    min_window_height,
                )));
                ctx.send_viewport_cmd(egui::ViewportCommand::MaxInnerSize(egui::vec2(
                    f32::INFINITY,
                    f32::INFINITY,
                )));
            } else {
                ctx.send_viewport_cmd(egui::ViewportCommand::MinInnerSize(egui::vec2(
                    main_width,
                    min_window_height,
                )));
                ctx.send_viewport_cmd(egui::ViewportCommand::MaxInnerSize(egui::vec2(
                    main_width,
                    f32::INFINITY,
                )));
            }
        }

        if (min_content_height.is_finite() && min_content_height > 0.0)
            || preview_toggled
            || lock_changed
        {
            let height_changed = self
                .last_fitted_height
                .is_none_or(|prev| (prev - min_window_height).abs() > 1.5);
            let current_size = ctx
                .input(|i| i.viewport().inner_rect.map(|r| r.size()))
                .unwrap_or(fallback_size);

            if preview_toggled
                || height_changed
                || current_size.y < min_window_height - 1.0
                || (!show_preview && (current_size.x - main_width).abs() > 1.0)
            {
                self.last_fitted_height = Some(min_window_height);
                if preview_toggled {
                    self.preview_panel_open = show_preview;
                }
                let target_width = if show_preview {
                    if preview_toggled {
                        main_width + default_preview_width
                    } else {
                        current_size.x.max(main_width + PREVIEW_MIN_WIDTH)
                    }
                } else {
                    main_width
                };
                let target_height = if height_changed {
                    min_window_height
                } else {
                    current_size.y.max(min_window_height)
                };

                if preview_toggled
                    || (current_size.y - target_height).abs() > 1.0
                    || (current_size.x - target_width).abs() > 1.0
                {
                    ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(egui::vec2(
                        target_width,
                        target_height,
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

fn render_preview_pane(
    ui: &mut egui::Ui,
    tex: Option<(egui::TextureId, u32, u32)>,
    view_mode: PreviewViewMode,
    current_exact_zoom_pct: f32,
    scroll_offset: &mut egui::Vec2,
    card_fill: Color32,
    border_color: Color32,
) -> Option<f32> {
    let mut rendered_zoom_pct: Option<f32> = None;
    let mut viewport_info: Option<(egui::Rect, egui::Vec2, u32, u32, f32)> = None;
    let mut dragged_pan = false;

    let frame_resp = Frame::new()
        .fill(card_fill)
        .stroke(Stroke::new(1.0, border_color))
        .corner_radius(CornerRadius::same(8))
        .inner_margin(Margin::same(8))
        .show(ui, |ui| {
            ui.set_width(ui.available_width());

            let Some((tex_id, img_w, img_h)) = tex else {
                ui.label(
                    RichText::new("Unavailable")
                        .size(12.0)
                        .color(ui.visuals().weak_text_color()),
                );
                return;
            };
            if img_w == 0 || img_h == 0 {
                return;
            }

            let avail = ui.available_size();
            let viewport_w = avail.x.max(40.0);
            let viewport_h = avail.y.max(120.0);
            let ppp = ui.ctx().pixels_per_point().max(1.0);

            let disp_size = match view_mode {
                PreviewViewMode::Framing(PreviewFraming::Fit) => {
                    let scale_x = viewport_w / (img_w as f32);
                    let scale_y = viewport_h / (img_h as f32);
                    let scale = scale_x.min(scale_y).max(0.01);
                    let disp_w = ((img_w as f32) * scale).max(1.0);
                    let disp_h = ((img_h as f32) * scale).max(1.0);
                    rendered_zoom_pct = Some((disp_w * ppp / (img_w as f32)) * 100.0);
                    egui::vec2(disp_w, disp_h)
                }
                PreviewViewMode::Framing(PreviewFraming::Fill) => {
                    let scale = (viewport_w / (img_w as f32))
                        .max(viewport_h / (img_h as f32))
                        .max(0.01);
                    let disp_w = ((img_w as f32) * scale).max(1.0);
                    rendered_zoom_pct = Some((disp_w * ppp / (img_w as f32)) * 100.0);
                    egui::vec2(disp_w, ((img_h as f32) * scale).max(1.0))
                }
                PreviewViewMode::Framing(PreviewFraming::Center) => {
                    rendered_zoom_pct = Some(100.0);
                    egui::vec2(
                        ((img_w as f32) / ppp).max(1.0),
                        ((img_h as f32) / ppp).max(1.0),
                    )
                }
                PreviewViewMode::FixedZoomPct(pct) => {
                    let effective_pct = if (current_exact_zoom_pct.round() as u16) == pct {
                        current_exact_zoom_pct.clamp(10.0, 500.0)
                    } else {
                        f32::from(pct.clamp(10, 500))
                    };
                    rendered_zoom_pct = Some(effective_pct);
                    let factor = effective_pct / 100.0;
                    egui::vec2(
                        (((img_w as f32) / ppp) * factor).max(1.0),
                        (((img_h as f32) / ppp) * factor).max(1.0),
                    )
                }
            };

            let max_scroll = egui::vec2(
                (disp_size.x - viewport_w).max(0.0),
                (disp_size.y - viewport_h).max(0.0),
            );
            let pannable = max_scroll.x > 0.5 || max_scroll.y > 0.5;
            *scroll_offset = scroll_offset.clamp(egui::Vec2::ZERO, max_scroll);

            let scroll_out = egui::ScrollArea::both()
                .id_salt("preview_scroll_single")
                .max_width(viewport_w)
                .max_height(viewport_h)
                .auto_shrink([false, false])
                .scroll_offset(*scroll_offset)
                .show(ui, |ui| {
                    let left_pad = ((viewport_w - disp_size.x) * 0.5).max(0.0);
                    let top_pad = ((viewport_h - disp_size.y) * 0.5).max(0.0);
                    let img_widget =
                        egui::Image::new((tex_id, disp_size)).sense(egui::Sense::click_and_drag());
                    let img_resp = if left_pad > 0.0 || top_pad > 0.0 {
                        let canvas_size = egui::vec2(
                            disp_size.x.max(viewport_w),
                            disp_size.y.max(viewport_h),
                        );
                        ui.allocate_ui_with_layout(
                            canvas_size,
                            egui::Layout::top_down(egui::Align::Min),
                            |ui| {
                                ui.add_space(top_pad);
                                ui.horizontal(|ui| {
                                    ui.add_space(left_pad);
                                    ui.add(img_widget)
                                })
                                .inner
                            },
                        )
                        .inner
                    } else {
                        ui.add(img_widget)
                    };

                    if pannable {
                        if img_resp.dragged_by(egui::PointerButton::Primary) {
                            *scroll_offset = (*scroll_offset - img_resp.drag_delta())
                                .clamp(egui::Vec2::ZERO, max_scroll);
                            dragged_pan = true;
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                        } else if img_resp.hovered() {
                            ui.ctx().set_cursor_icon(egui::CursorIcon::Grab);
                        }
                    }
                });

            if !dragged_pan {
                *scroll_offset = scroll_out
                    .state
                    .clamped_offset()
                    .clamp(egui::Vec2::ZERO, max_scroll);
            }
            viewport_info = Some((scroll_out.inner_rect, disp_size, img_w, img_h, ppp));
        });

    let base_zoom = rendered_zoom_pct?;
    let (viewport_rect, disp_size, img_w, img_h, ppp) = viewport_info?;

    let (zoom_delta, scroll_y, ctrl_held, pointer_pos) = ui.ctx().input(|i| {
        let raw_wheel_y: f32 = i
            .raw
            .events
            .iter()
            .filter_map(|ev| match ev {
                egui::Event::MouseWheel { unit, delta, .. } => {
                    let px = match unit {
                        egui::MouseWheelUnit::Point => delta.y,
                        egui::MouseWheelUnit::Line => delta.y * 24.0,
                        egui::MouseWheelUnit::Page => delta.y * 120.0,
                    };
                    Some(px)
                }
                _ => None,
            })
            .sum();
        (
            i.zoom_delta(),
            raw_wheel_y,
            i.modifiers.ctrl || i.modifiers.command,
            i.pointer.hover_pos(),
        )
    });
    let hovered = pointer_pos.is_some_and(|p| viewport_rect.contains(p))
        || ui.rect_contains_pointer(frame_resp.response.rect);
    if !hovered {
        return None;
    }

    let factor = if (zoom_delta - 1.0).abs() > 1e-4 {
        zoom_delta
    } else if ctrl_held && scroll_y.abs() > 0.0 {
        1.1_f32.powf(scroll_y / 40.0)
    } else {
        1.0
    };

    if (factor - 1.0).abs() > 1e-4 && factor.is_finite() && factor > 0.0 {
        let new_exact_pct = (base_zoom * factor).clamp(10.0, 500.0);
        let old_zoom = (base_zoom / 100.0).max(0.01);
        let new_zoom = (new_exact_pct / 100.0).max(0.01);
        let cursor_rel = pointer_pos
            .map(|p| p - viewport_rect.min)
            .unwrap_or_else(|| viewport_rect.size() * 0.5);
        let viewport_size = viewport_rect.size();
        let old_pad = egui::vec2(
            ((viewport_size.x - disp_size.x) * 0.5).max(0.0),
            ((viewport_size.y - disp_size.y) * 0.5).max(0.0),
        );
        let content_pos = (*scroll_offset + cursor_rel - old_pad) / old_zoom;
        let new_disp_w = ((img_w as f32) / ppp) * new_zoom;
        let new_disp_h = ((img_h as f32) / ppp) * new_zoom;
        let new_pad = egui::vec2(
            ((viewport_size.x - new_disp_w) * 0.5).max(0.0),
            ((viewport_size.y - new_disp_h) * 0.5).max(0.0),
        );
        let new_max_scroll = egui::vec2(
            (new_disp_w - viewport_size.x).max(0.0),
            (new_disp_h - viewport_size.y).max(0.0),
        );
        *scroll_offset = (content_pos * new_zoom + new_pad - cursor_rel)
            .clamp(egui::Vec2::ZERO, new_max_scroll);
        Some(new_exact_pct)
    } else {
        None
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
                && delta.is_some_and(|d| {
                    !(d.format_changed || d.color_depth_changed || d.filter_changed)
                });

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
                            || d.color_depth_changed
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

fn sanitize_filename_suffix(raw_suffix: &str) -> String {
    let cleaned: String = raw_suffix
        .chars()
        .filter(|&c| {
            !c.is_control()
                && !matches!(c, '\\' | '/' | ':' | '*' | '?' | '"' | '<' | '>' | '|')
        })
        .collect();
    cleaned.trim_end_matches([' ', '.']).to_string()
}

fn build_output_path(input_path: &Path, raw_suffix: &str) -> PathBuf {
    let stem = input_path
        .file_stem()
        .and_then(|s| s.to_str())
        .filter(|s| !s.is_empty())
        .unwrap_or("output");
    let sanitized = sanitize_filename_suffix(raw_suffix);
    if sanitized.is_empty() {
        input_path.with_file_name(format!("{stem}.pdf"))
    } else {
        input_path.with_file_name(format!("{stem}{sanitized}.pdf"))
    }
}

fn is_file_locked_error(err: &std::io::Error) -> bool {
    err.kind() == std::io::ErrorKind::PermissionDenied
        || matches!(err.raw_os_error(), Some(5 | 32 | 33))
}

#[cold]
fn format_file_save_error(dest_path: &Path, err: &std::io::Error) -> String {
    if is_file_locked_error(err) {
        format!("File is open in another program: {}", dest_path.display())
    } else {
        format!("Failed to save {}: {err}", dest_path.display())
    }
}

#[cfg(test)]
fn write_pdf_bytes_safely(dest_path: &Path, bytes: &[u8]) -> Result<(), String> {
    write_pdf_bytes_safely_indexed(dest_path, bytes, 0)
}

/// Always writes `bytes` to a unique temporary file in the destination folder via `BufWriter`,
/// flushes and drops all file handles, and atomically replaces `dest_path` so crashes or
/// Windows file locks never corrupt the target PDF.
fn write_pdf_bytes_safely_indexed(
    dest_path: &Path,
    bytes: &[u8],
    job_index: usize,
) -> Result<(), String> {
    use std::io::{BufWriter, Write};

    let parent = dest_path
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or_else(|| Path::new("."));
    let pid = std::process::id();
    let mut tmp_path = parent.join(format!(".tmp_pdfcomp_{pid}_{job_index}.pdf"));
    let mut counter = 0usize;
    while tmp_path.exists() {
        counter += 1;
        tmp_path = parent.join(format!(".tmp_pdfcomp_{pid}_{job_index}_{counter}.pdf"));
    }

    {
        let file = std::fs::File::create(&tmp_path)
            .map_err(|e| format_file_save_error(&tmp_path, &e))?;
        let mut writer = BufWriter::new(file);
        if let Err(e) = writer.write_all(bytes).and_then(|_| writer.flush()) {
            drop(writer);
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format_file_save_error(&tmp_path, &e));
        }
        let file = match writer.into_inner() {
            Ok(f) => f,
            Err(into_err) => {
                let err = into_err.into_error();
                let _ = std::fs::remove_file(&tmp_path);
                return Err(format_file_save_error(&tmp_path, &err));
            }
        };
        if let Err(e) = file.sync_all() {
            drop(file);
            let _ = std::fs::remove_file(&tmp_path);
            return Err(format_file_save_error(&tmp_path, &e));
        }
        drop(file);
    }

    match std::fs::rename(&tmp_path, dest_path) {
        Ok(()) => Ok(()),
        Err(first_err) => {
            if dest_path.exists() {
                match std::fs::remove_file(dest_path) {
                    Ok(()) => match std::fs::rename(&tmp_path, dest_path) {
                        Ok(()) => return Ok(()),
                        Err(rename_err) => {
                            let _ = std::fs::remove_file(&tmp_path);
                            return Err(format_file_save_error(dest_path, &rename_err));
                        }
                    },
                    Err(remove_err) => {
                        let _ = std::fs::remove_file(&tmp_path);
                        return Err(format_file_save_error(dest_path, &remove_err));
                    }
                }
            }
            let _ = std::fs::remove_file(&tmp_path);
            Err(format_file_save_error(dest_path, &first_err))
        }
    }
}

fn unique_batch_output_paths(files: &[PdfFileItem], folder: &Path) -> Vec<PathBuf> {
    let mut used = HashSet::with_capacity(files.len());
    let mut paths = Vec::with_capacity(files.len());

    for item in files {
        let source_path = item
            .path
            .as_deref()
            .unwrap_or_else(|| Path::new(&item.file_name));
        let base_out = build_output_path(source_path, &item.settings.suffix);
        let base_name = base_out
            .file_name()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("output.pdf")
            .to_string();
        let base_stem = base_out
            .file_stem()
            .and_then(|s| s.to_str())
            .filter(|s| !s.is_empty())
            .unwrap_or("output")
            .to_string();

        let mut candidate = base_name;
        let mut n = 2usize;
        while !used.insert(candidate.to_lowercase()) {
            candidate = format!("{base_stem}_{n}.pdf");
            n += 1;
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

#[cfg(test)]
fn split_row_columns<const N: usize>(row_rect: egui::Rect, widths: [f32; N]) -> [egui::Rect; N] {
    let mut rects = [egui::Rect::NOTHING; N];
    split_row_columns_into(row_rect, &widths, &mut rects);
    rects
}

fn split_row_columns_into(row_rect: egui::Rect, widths: &[f32], out: &mut [egui::Rect]) {
    let n = out.len().min(widths.len());
    let y0 = row_rect.min.y;
    let y1 = row_rect.max.y;
    let mut x = row_rect.min.x;
    for (idx, rect) in out[..n].iter_mut().enumerate() {
        let next_x = if idx + 1 == n {
            row_rect.max.x
        } else {
            (x + widths[idx]).min(row_rect.max.x)
        };
        *rect = egui::Rect::from_min_max(egui::pos2(x, y0), egui::pos2(next_x, y1));
        x = next_x;
    }
}

#[inline]
fn render_table_header_row<const N: usize>(
    ui: &mut egui::Ui,
    total_w: f32,
    widths: [f32; N],
    aligns: [egui::Align; N],
    titles: [&str; N],
) {
    let _ = render_table_header_row_interactive_slice(ui, total_w, &widths, &aligns, &titles, None);
}

#[inline]
fn render_table_header_row_interactive<const N: usize>(
    ui: &mut egui::Ui,
    total_w: f32,
    widths: [f32; N],
    aligns: [egui::Align; N],
    titles: [&str; N],
    interactive_col: Option<(usize, egui::Id, &str)>,
) -> bool {
    render_table_header_row_interactive_slice(
        ui,
        total_w,
        &widths,
        &aligns,
        &titles,
        interactive_col,
    )
}

fn render_table_header_row_interactive_slice(
    ui: &mut egui::Ui,
    total_w: f32,
    widths: &[f32],
    aligns: &[egui::Align],
    titles: &[&str],
    interactive_col: Option<(usize, egui::Id, &str)>,
) -> bool {
    let dark = ui.visuals().dark_mode;
    let divider_color = table_header_divider_color(dark);
    let (header_rect, _) =
        ui.allocate_exact_size(egui::vec2(total_w, TABLE_HEADER_HEIGHT), egui::Sense::hover());

    let mut col_rects: SmallVec<[egui::Rect; 8]> = smallvec::smallvec![egui::Rect::NOTHING; widths.len()];
    split_row_columns_into(header_rect, widths, &mut col_rects);

    let mut clicked = false;
    let mut hovered_col_rect: Option<egui::Rect> = None;
    if let Some((col_idx, interact_id, tooltip)) = interactive_col {
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

    let weak_color = Some(ui.visuals().weak_text_color());
    let colors: SmallVec<[Option<Color32>; 8]> = smallvec::smallvec![weak_color; widths.len()];
    let strong: SmallVec<[bool; 8]> = smallvec::smallvec![true; widths.len()];
    paint_row_cells_slice(
        ui,
        header_rect,
        &TableRowCellsSlice {
            widths,
            aligns,
            texts: titles,
            colors: &colors,
            strong: &strong,
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

struct TableRowCellsSlice<'a> {
    widths: &'a [f32],
    aligns: &'a [egui::Align],
    texts: &'a [&'a str],
    colors: &'a [Option<Color32>],
    strong: &'a [bool],
    font_size: f32,
    divider_stroke: Stroke,
    divider_v_inset: f32,
}

#[inline]
fn paint_row_cells<const N: usize>(
    ui: &egui::Ui,
    row_rect: egui::Rect,
    cells: &TableRowCells<'_, N>,
) {
    paint_row_cells_slice(
        ui,
        row_rect,
        &TableRowCellsSlice {
            widths: &cells.widths,
            aligns: &cells.aligns,
            texts: &cells.texts,
            colors: &cells.colors,
            strong: &cells.strong,
            font_size: cells.font_size,
            divider_stroke: cells.divider_stroke,
            divider_v_inset: cells.divider_v_inset,
        },
    );
}

fn paint_row_cells_slice(
    ui: &egui::Ui,
    row_rect: egui::Rect,
    cells: &TableRowCellsSlice<'_>,
) {
    let n = cells.widths.len();
    let mut col_rects: SmallVec<[egui::Rect; 8]> = smallvec::smallvec![egui::Rect::NOTHING; n];
    split_row_columns_into(row_rect, cells.widths, &mut col_rects);
    let painter = ui.painter();
    let div_y0 = row_rect.min.y + cells.divider_v_inset;
    let div_y1 = row_rect.max.y - cells.divider_v_inset;
    col_rects[..n.saturating_sub(1)].iter().for_each(|col| {
        painter.vline(col.max.x, div_y0..=div_y1, cells.divider_stroke);
    });

    let fallback_color = ui.visuals().text_color();
    for (col, &col_rect) in col_rects.iter().enumerate() {
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

fn format_saved_pct(saved_pct: f64) -> String {
    if saved_pct >= 0.0 {
        format!("{saved_pct:.1}%")
    } else {
        format!("+{:.1}% larger", saved_pct.abs())
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

    const ACTIVE_DEFAULT_WINDOW_WIDTH: f32 = DEFAULT_WINDOW_WIDTH + PREVIEW_DEFAULT_WIDTH;
    const ACTIVE_MIN_WINDOW_WIDTH: f32 = MIN_WINDOW_WIDTH + PREVIEW_DEFAULT_WIDTH;

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
            source_id: next_file_source_id(),
            path: Some(PathBuf::from(format!(r"C:\docs\{name}"))),
            file_name: name.to_string(),
            original_size_mb: bytes_to_mb(file_bytes),
            in_memory_bytes: None,
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

    fn has_viewport_bounds_cmds(output: &egui::FullOutput) -> (Option<egui::Vec2>, Option<egui::Vec2>) {
        let mut min_sz = None;
        let mut max_sz = None;
        if let Some(vo) = output.viewport_output.get(&egui::ViewportId::ROOT) {
            for cmd in &vo.commands {
                match cmd {
                    egui::ViewportCommand::MinInnerSize(sz) => min_sz = Some(*sz),
                    egui::ViewportCommand::MaxInnerSize(sz) => max_sz = Some(*sz),
                    _ => {}
                }
            }
        }
        (min_sz, max_sz)
    }

    #[test]
    fn tight_viewport_height_fits_before_and_after_save_without_churn() {
        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();

        // DPI-scaled widths default to 1920pt fallback (27.5% = 528, 42.5% = 816, total = 70% = 1344).
        assert_eq!(dpi_scaled_widths(&ctx), (DEFAULT_WINDOW_WIDTH, PREVIEW_DEFAULT_WIDTH));

        // Initial launch with 0 files: Compression Summary is hidden, horizontal width locked to main_width.
        let out1 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        let (min1, max1) = has_viewport_bounds_cmds(&out1);
        let initial_fitted = app
            .last_fitted_height
            .expect("initial height should be measured");
        assert_eq!(min1, Some(egui::vec2(DEFAULT_WINDOW_WIDTH, initial_fitted)));
        assert_eq!(max1, Some(egui::vec2(DEFAULT_WINDOW_WIDTH, f32::INFINITY)));
        assert!(
            (320.0..=420.0).contains(&initial_fitted),
            "expected tight initial height without Compression Summary, got {initial_fitted}"
        );

        // Next frame at fitted height emits no further InnerSize command (no jitter/loop)
        let out2 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, initial_fitted);
        assert_eq!(has_inner_size_cmd(&out2), None);

        // Simulate single-file completed save ("Saved to: ..." appears in Compression Summary)
        let mut item = make_test_item("homework.pdf", 4 * 1024 * 1024);
        item.result = Some(Box::new(CompressionResult::from_stats(
            PathBuf::from(r"C:\Users\demo\homework-compressed.pdf"),
            CompressionStats {
                original_bytes: 4 * 1024 * 1024,
                compressed_bytes: 1_200_000,
                ..Default::default()
            },
            &item.settings,
            item.original_size_mb,
        )));
        app.files = vec![item];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);
        app.show_preview = true;

        let out3 = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, initial_fitted);
        let post_save_fitted = app.last_fitted_height.expect("post-save height measured");
        assert!(
            post_save_fitted >= initial_fitted,
            "window height should accommodate 'Saved to: ...' and preview ({initial_fitted} -> {post_save_fitted})"
        );
        let requested = has_inner_size_cmd(&out3).expect("should emit InnerSize when preview opens");
        assert!((requested.x - ACTIVE_DEFAULT_WINDOW_WIDTH).abs() < 0.5);
        assert!((requested.y - post_save_fitted).abs() < 0.5);
        let (min3, max3) = has_viewport_bounds_cmds(&out3);
        assert_eq!(min3, Some(egui::vec2(DEFAULT_WINDOW_WIDTH + PREVIEW_MIN_WIDTH, post_save_fitted)));
        assert_eq!(max3, Some(egui::vec2(f32::INFINITY, f32::INFINITY)));

        // Subsequent frame at expanded preview size settles with no further resize
        let out4 = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            post_save_fitted,
        );
        assert_eq!(has_inner_size_cmd(&out4), None);

        // Clearing files collapses preview side panel, restores main window width, and locks horizontal resize
        app.clear_files();
        let out_collapse = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            post_save_fitted,
        );
        let collapsed = has_inner_size_cmd(&out_collapse)
            .expect("clearing files should collapse window back to default width");
        assert!((collapsed.x - DEFAULT_WINDOW_WIDTH).abs() < 0.5);
        let (min_col, max_col) = has_viewport_bounds_cmds(&out_collapse);
        assert_eq!(min_col, Some(egui::vec2(DEFAULT_WINDOW_WIDTH, initial_fitted)));
        assert_eq!(max_col, Some(egui::vec2(DEFAULT_WINDOW_WIDTH, f32::INFINITY)));

        // Narrowest supported active width (528px locked main + preview) fits cleanly without overflow
        let compact_ctx = egui::Context::default();
        let mut compact_item = make_test_item("homework_scan_2026.pdf", 3_500_000);
        compact_item.settings.options.format = ImageFormatChoice::Png;
        compact_item.settings.target_mode = TargetMode::Megabytes;
        let mut compact_app = PdfCompressorApp {
            show_preview: true,
            files: vec![compact_item],
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };
        let _out5 = run_frame(
            &compact_ctx,
            &mut compact_app,
            ACTIVE_MIN_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        let compact_height = compact_app
            .last_fitted_height
            .expect("compact height should be measured");
        assert!(
            (MIN_PREVIEW_WINDOW_HEIGHT..=600.0).contains(&compact_height),
            "expected compact active layout at {ACTIVE_MIN_WINDOW_WIDTH}px width, got {compact_height}"
        );
        assert!(
            compact_app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "expected controls at {ACTIVE_MIN_WINDOW_WIDTH}px window width to fit inside {}px central width without horizontal overflow, got {}px",
            MIN_WINDOW_WIDTH - 20.0,
            compact_app.last_content_width
        );
    }

    #[test]
    fn estimated_size_and_summary_metrics_follow_state_transitions() {
        let mut app = PdfCompressorApp::default();

        // Unloaded: no files
        assert!(app.files.is_empty());

        // Loaded before compression: inline estimate has no tilde and summary shows muted estimate
        let item = make_test_item("notes.pdf", 4_000_000);
        let inline_est = estimated_size_text(&item);
        assert!(inline_est.starts_with("Estimated Size: "));
        assert!(!inline_est.contains('~'));
        assert!(!inline_est.contains('%'));

        app.files = vec![item];
        app.selected_indices.insert(0);
        app.selection_anchor = Some(0);
        let est = app.files[0].estimated_compressed_bytes();
        assert!(est > 0);
        assert_eq!(ImageFormatChoice::Png.label(), "PNG");
        assert!(app.files[0].estimated_saved_percentage().is_some());

        // After compression: actual new size & saved percentage are populated
        let orig_mb = app.files[0].original_size_mb;
        app.files[0].result = Some(Box::new(CompressionResult::from_stats(
            PathBuf::from("notes-compressed.pdf"),
            CompressionStats {
                original_bytes: 4_000_000,
                compressed_bytes: 1_156_000,
                ..Default::default()
            },
            &app.files[0].settings,
            orig_mb,
        )));
        let res = app.files[0].result.as_ref().unwrap();
        assert_eq!(format_saved_pct(res.space_saved_pct), "71.1%");
    }

    #[test]
    fn multi_select_click_ctrl_shift_and_ctrl_a_behave_accurately() {
        let mut app = PdfCompressorApp {
            files: (1..=6)
                .map(|i| make_test_item(&format!("page_{i}.pdf"), i * 1_000_000))
                .collect(),
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };

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
        let mut app = PdfCompressorApp {
            files: vec![
                make_test_item("small.pdf", 1_000_000),   // ~0.95 MB
                make_test_item("medium.pdf", 4_000_000),  // ~3.81 MB
                make_test_item("same\\small.pdf", 8_000_000),
            ],
            selected_indices: BTreeSet::from([1]),
            selection_anchor: Some(1),
            ..Default::default()
        };

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

        // Unique batch output paths preserve <stem><suffix>.pdf and disambiguate collisions
        let out_dir = PathBuf::from(r"C:\out");
        let paths = unique_batch_output_paths(&app.files, &out_dir);
        assert_eq!(paths[0], out_dir.join("small-compressed.pdf"));
        assert_eq!(paths[1], out_dir.join("medium-compressed.pdf"));
        assert_ne!(paths[2], paths[0]);

        // Empty suffix saves with exact original filename, and write_pdf_bytes_safely overwrites atomically
        app.files[0].settings.suffix.clear();
        let paths_empty = unique_batch_output_paths(&app.files, &out_dir);
        assert_eq!(paths_empty[0], out_dir.join("small.pdf"));

        let temp_dir = std::env::temp_dir().join(format!(
            "pdf_comp_atomic_save_test_{}",
            std::process::id()
        ));
        let _ = std::fs::create_dir_all(&temp_dir);
        let target_file = temp_dir.join("in_place.pdf");
        std::fs::write(&target_file, b"%PDF-old").expect("seed existing file");
        write_pdf_bytes_safely(&target_file, b"%PDF-new-compressed").expect("safe overwrite");
        assert_eq!(
            std::fs::read(&target_file).expect("read overwritten file"),
            b"%PDF-new-compressed"
        );
        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn batch_ui_renders_cleanly_at_min_width_and_caps_scroll_for_many_files() {
        let ctx = egui::Context::default();
        let mut app_small_batch = PdfCompressorApp {
            files: (1..=3)
                .map(|i| make_test_item(&format!("doc_{i}.pdf"), (i + 1) * 1_500_000))
                .collect(),
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };
        app_small_batch.files[0].settings.target_mode = TargetMode::Megabytes;
        app_small_batch.files[0].settings.target_size_mb = 1.0;
        app_small_batch.files[0].result = Some(Box::new(CompressionResult::from_stats(
            PathBuf::from(r"C:\out\doc_1_compressed.pdf"),
            CompressionStats {
                original_bytes: 3_000_000,
                compressed_bytes: 1_500_000,
                ..Default::default()
            },
            &app_small_batch.files[0].settings,
            app_small_batch.files[0].original_size_mb,
        )));
        assert!(app_small_batch.files[0].result.as_ref().unwrap().over_target);

        app_small_batch.files[1].result = Some(Box::new(CompressionResult::from_stats(
            PathBuf::from(r"C:\out\doc_2_compressed.pdf"),
            CompressionStats {
                original_bytes: 4_500_000,
                compressed_bytes: 1_200_000,
                ..Default::default()
            },
            &app_small_batch.files[1].settings,
            app_small_batch.files[1].original_size_mb,
        )));
        assert!(!app_small_batch.files[1].result.as_ref().unwrap().over_target);
        app_small_batch.batch_saved_folder = Some(PathBuf::from(r"C:\out"));
        app_small_batch.show_preview = true;

        let _ = run_frame(&ctx, &mut app_small_batch, ACTIVE_MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app_small_batch.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "multi-file UI overflowed {MIN_WINDOW_WIDTH}px central width: {}px",
            app_small_batch.last_content_width
        );
        let three_file_h = app_small_batch.last_fitted_height.unwrap();

        let mut app_large_batch = PdfCompressorApp {
            show_preview: true,
            files: (1..=15)
                .map(|i| make_test_item(&format!("scan_{i:02}.pdf"), 2_500_000))
                .collect(),
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };
        let _ = run_frame(&ctx, &mut app_large_batch, ACTIVE_DEFAULT_WINDOW_WIDTH, 900.0);
        let fifteen_file_h = app_large_batch.last_fitted_height.unwrap();
        assert!(
            fifteen_file_h > three_file_h && fifteen_file_h < 950.0,
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
        let mut app = PdfCompressorApp {
            files: vec![
                make_test_item("short.pdf", 1_200_000),
                make_test_item(
                    "very_long_scanned_homework_document_filename_2026_semester_final_submission_v3.pdf",
                    2_400_000,
                ),
            ],
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            show_preview: true,
            ..Default::default()
        };

        let _ = run_frame(&ctx, &mut app, ACTIVE_MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "long filenames should truncate instead of overflowing {MIN_WINDOW_WIDTH}px central width: {}px",
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
        let mut app = PdfCompressorApp {
            files: vec![make_test_item("one.pdf", 2_000_000)],
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };

        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
        let jpeg_h = app.last_fitted_height.unwrap();

        app.files[0].settings.options.format = ImageFormatChoice::Jpeg2000;
        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
        let jpx_h = app.last_fitted_height.unwrap();

        app.files[0].settings.options.format = ImageFormatChoice::Png;
        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
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
        app.show_preview = true;
        app.files[0].result = Some(Box::new(under));
        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
        let under_h = app.last_fitted_height.unwrap();

        app.files[0].result = Some(Box::new(over));
        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
        let over_h = app.last_fitted_height.unwrap();
        assert!(
            (under_h - over_h).abs() < 1.0,
            "Over-target summary should have no extra warning text height ({under_h} vs {over_h})"
        );
        app.files[0].result = None;

        // Percent mode UI renders cleanly at ACTIVE_MIN_WINDOW_WIDTH without horizontal overflow
        let _ = run_frame(&ctx, &mut app, ACTIVE_MIN_WINDOW_WIDTH, 700.0);
        assert!(
            app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5,
            "Percent mode UI overflowed {MIN_WINDOW_WIDTH}px central width: {}px",
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
        let _ = run_frame(&ctx, &mut app, ACTIVE_DEFAULT_WINDOW_WIDTH, 700.0);
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
        let mut app = PdfCompressorApp {
            show_preview: true,
            files: vec![make_test_item("single.pdf", 4_000_000)],
            selected_indices: BTreeSet::from([0]),
            selection_anchor: Some(0),
            ..Default::default()
        };

        // Two frames let egui::Grid settle its shared column width cleanly at ACTIVE_MIN_WINDOW_WIDTH.
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_MIN_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_MIN_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );

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

        // Switching Target Size to Megabytes must NOT shift or resize the sliders.
        app.files[0].settings.target_mode = TargetMode::Megabytes;
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_MIN_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        let q_slider_mb = app.last_quality_slider_rect.expect("q slider mb");
        let q_drag_mb = app.last_quality_drag_rect.expect("q drag mb");
        assert!(
            (q_slider_mb.min.x - q_slider.min.x).abs() < 0.5
                && (q_slider_mb.width() - q_slider.width()).abs() < 0.5
                && (q_drag_mb.width() - q_drag.width()).abs() < 0.5,
            "sliders must stay fixed when Target Size mode changes: off={q_slider:?}, mb={q_slider_mb:?}"
        );
        assert!(app.last_content_width <= MIN_WINDOW_WIDTH - 20.0 + 0.5);

        // Busy state renders cleanly with compact progress counter (single 1/1 and batch 2/3)
        let (_tx, rx) = mpsc::channel();
        app.worker_rx = Some(rx);
        app.progress = Some(ProgressUpdate {
            current: 1,
            total: 1,
        });
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_MIN_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
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
        let _ = run_frame(&ctx, &mut app, ACTIVE_MIN_WINDOW_WIDTH, 700.0);
        assert!(app.is_busy());

        app.worker_rx = None;
        app.progress = None;
        let _ = run_frame(&ctx, &mut app, ACTIVE_MIN_WINDOW_WIDTH, 700.0);
        assert!(!app.is_busy());
    }

    #[test]
    fn preview_panel_visibility_page_eviction_and_clear_behave_cleanly() {
        use image::{DynamicImage, ImageBuffer, Rgb};
        use lopdf::{dictionary, Document, Object, Stream};

        fn build_two_page_preview_pdf() -> Vec<u8> {
            let mut doc = Document::with_version("1.5");
            let pages_id = doc.new_object_id();

            let img1 = DynamicImage::ImageRgb8(ImageBuffer::from_fn(24, 24, |x, y| {
                Rgb([(x * 10) as u8, (y * 10) as u8, 180])
            }));
            let raw1 = img1.to_rgb8().into_raw();
            let img1_id = doc.add_object(Stream::new(
                dictionary! {
                    "Type" => "XObject",
                    "Subtype" => "Image",
                    "Width" => 24,
                    "Height" => 24,
                    "ColorSpace" => "DeviceRGB",
                    "BitsPerComponent" => 8,
                },
                raw1,
            ));

            let content1_id = doc.add_object(Stream::new(
                dictionary! {},
                b"q 24 0 0 24 0 0 cm /Im1 Do Q".to_vec(),
            ));
            let page1_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 200.into(), 200.into()],
                "Resources" => dictionary! {
                    "XObject" => dictionary! { "Im1" => img1_id },
                },
                "Contents" => content1_id,
            });
            let content2_id = doc.add_object(Stream::new(
                dictionary! {},
                b"0.15 0.15 0.15 rg 20 20 80 80 re f".to_vec(),
            ));
            let page2_id = doc.add_object(dictionary! {
                "Type" => "Page",
                "Parent" => pages_id,
                "MediaBox" => vec![0.into(), 0.into(), 200.into(), 200.into()],
                "Contents" => content2_id,
            });
            doc.objects.insert(
                pages_id,
                Object::Dictionary(dictionary! {
                    "Type" => "Pages",
                    "Kids" => vec![page1_id.into(), page2_id.into()],
                    "Count" => 2,
                }),
            );
            let catalog_id = doc.add_object(dictionary! {
                "Type" => "Catalog",
                "Pages" => pages_id,
            });
            doc.trailer.set("Root", catalog_id);
            let mut out = Vec::new();
            doc.save_to(&mut out).unwrap();
            out
        }

        let ctx = egui::Context::default();
        let mut app = PdfCompressorApp::default();
        assert!(!app.show_preview, "Show Preview must default to unchecked");
        assert_eq!(app.preview_framing, PreviewFraming::Fit);

        // Empty state hides preview panel completely.
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        assert!(!app.preview_panel_open);
        assert!(app.preview_display.is_none());

        // Loading a 2-page PDF keeps preview panel hidden until user checks "Show Preview".
        let pdf_bytes = build_two_page_preview_pdf();
        app.load_pdf_inputs(vec![(None, "two_page.pdf".to_string(), pdf_bytes)], vec![]);
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        assert!(!app.preview_panel_open);
        assert!(app.preview_display.is_none());

        // Checking "Show Preview" opens preview panel and loads full-page Page 1 preview (200pt * 2.0 = 400px).
        app.show_preview = true;
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        assert!(app.preview_panel_open);
        for _ in 0..40 {
            let _ = run_frame(
                &ctx,
                &mut app,
                ACTIVE_DEFAULT_WINDOW_WIDTH,
                MIN_PREVIEW_WINDOW_HEIGHT,
            );
            if app.preview_display.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let page1_display = app
            .preview_display
            .as_ref()
            .expect("page 1 preview should finish loading");
        assert!(!page1_display.synthetic_empty);
        assert_eq!((page1_display.orig_width, page1_display.orig_height), (400, 400));
        assert!(page1_display.orig_texture.is_some());

        // Unchecking "Show Preview" hides side panel and evicts preview textures immediately.
        app.show_preview = false;
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, MIN_PREVIEW_WINDOW_HEIGHT);
        assert!(!app.preview_panel_open);
        assert!(app.preview_display.is_none());

        // Re-checking "Show Preview" restores the side panel and verifies capsule page edit clamping.
        app.show_preview = true;
        app.preview_page_edit_buf = "99".to_string();
        app.commit_preview_page_edit(2);
        assert_eq!(app.preview_page, 2);
        app.preview_page = 1;
        app.preview_page_edit_buf = "1".to_string();
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        assert!(app.preview_panel_open);

        // Switching to Page 2 (vector-only) immediately evicts Page 1 textures and renders Page 2 full page.
        app.preview_page = 2;
        app.sync_preview_selection_and_poll(&ctx);
        assert!(
            app.preview_display.is_none(),
            "previous page textures must be evicted immediately on page change"
        );
        for _ in 0..40 {
            let _ = run_frame(
                &ctx,
                &mut app,
                ACTIVE_DEFAULT_WINDOW_WIDTH,
                MIN_PREVIEW_WINDOW_HEIGHT,
            );
            if app.preview_display.is_some() {
                break;
            }
            thread::sleep(Duration::from_millis(10));
        }
        let page2_display = app
            .preview_display
            .as_ref()
            .expect("page 2 vector full-page preview should load");
        assert!(!page2_display.synthetic_empty);
        assert_eq!((page2_display.orig_width, page2_display.orig_height), (400, 400));
        assert!(page2_display.orig_texture.is_some());

        // Verify framing cycle Fit -> Fill -> Center -> Fit and clamped custom zoom input (10%..=500%).
        assert_eq!(app.preview_framing.next(), PreviewFraming::Fill);
        assert_eq!(PreviewFraming::Fill.next(), PreviewFraming::Center);
        assert_eq!(PreviewFraming::Center.next(), PreviewFraming::Fit);

        app.preview_zoom_edit_buf = "750".to_string();
        app.commit_preview_zoom_edit();
        assert_eq!(app.active_preview_view_mode(), PreviewViewMode::FixedZoomPct(500));

        app.preview_zoom_edit_buf = "4%".to_string();
        app.commit_preview_zoom_edit();
        assert_eq!(app.active_preview_view_mode(), PreviewViewMode::FixedZoomPct(10));

        app.preview_fixed_zoom = false;
        app.preview_framing = PreviewFraming::Fill;
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        app.preview_framing = PreviewFraming::Center;
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        app.preview_show_original_held = true;
        let _ = run_frame(
            &ctx,
            &mut app,
            ACTIVE_DEFAULT_WINDOW_WIDTH,
            MIN_PREVIEW_WINDOW_HEIGHT,
        );
        assert!(!app.preview_show_original_held);

        // Clearing files drops textures and collapses the preview panel.
        app.clear_files();
        assert!(app.preview_display.is_none());
        let _ = run_frame(&ctx, &mut app, DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT);
        assert!(!app.preview_panel_open);

        // Changing Color-depth updates PreviewKey, estimates, and auto-falls back on format switch.
        let mut cd_app = PdfCompressorApp::default();
        cd_app.files.push(PdfCompressorApp::build_loaded_item(
            None,
            "depth.pdf".to_string(),
            None,
            PdfSummary::synthetic_for_test(2, 4_000_000, 120_000, 4, 1000, 800_000, false),
        ));
        cd_app.selected_indices.insert(0);
        assert_eq!(cd_app.files[0].settings.options.color_depth, ColorDepth::TrueColor24);

        let est_24 = cd_app.files[0].cached_estimated_bytes;
        let mut draft = cd_app.files[0].settings.clone();
        draft.options.format = ImageFormatChoice::Jpeg2000;
        draft.options.color_depth = ColorDepth::Hdr30;
        cd_app.apply_settings_delta(
            &draft,
            SettingsDelta {
                format_changed: true,
                color_depth_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(cd_app.files[0].settings.options.color_depth, ColorDepth::Hdr30);
        assert_ne!(cd_app.files[0].cached_estimated_bytes, est_24);

        let mut switch_back = cd_app.files[0].settings.clone();
        switch_back.options.format = ImageFormatChoice::Jpeg;
        cd_app.apply_settings_delta(
            &switch_back,
            SettingsDelta {
                format_changed: true,
                ..Default::default()
            },
        );
        assert_eq!(
            cd_app.files[0].settings.options.color_depth,
            ColorDepth::TrueColor24,
            "switching from JPEG 2000 HDR 30-bit to JPG must auto-fallback to True Color (24-bit)"
        );
    }

    #[test]
    fn build_output_path_sanitizes_suffix_and_preserves_original_when_blank() {
        let input = Path::new(r"C:\docs\report.pdf");
        assert_eq!(
            build_output_path(input, "-compressed"),
            PathBuf::from(r"C:\docs\report-compressed.pdf")
        );
        assert_eq!(
            build_output_path(input, ""),
            PathBuf::from(r"C:\docs\report.pdf")
        );
        // Forbidden Windows chars + ASCII controls + trailing spaces/dots are stripped cleanly.
        assert_eq!(
            build_output_path(input, "-v2:*/?\\<>|\x07 . . "),
            PathBuf::from(r"C:\docs\report-v2.pdf")
        );
        // Suffix containing only forbidden chars, dots, or spaces collapses to exact original name.
        assert_eq!(
            build_output_path(input, " :*?\"<>| ...  "),
            PathBuf::from(r"C:\docs\report.pdf")
        );
    }
}
