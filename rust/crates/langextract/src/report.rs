//! Per-document and per-chunk observability reports.
//!
//! Every call to [`extract`](crate::extract) (and every document in a
//! batch run) produces a [`DocumentReport`] alongside the grounded
//! [`AnnotatedDocument`](langextract_core::AnnotatedDocument). The
//! report captures every statistic the pipeline observed while
//! processing the document, which is the quality-gate signal your
//! downstream stages need to decide whether a document is safe to
//! promote into the rule-IR layer.
//!
//! # Signals captured
//!
//! - **Grounding rate** — fraction of emitted extractions that
//!   landed with a `char_interval`.
//! - **Alignment status breakdown** — Exact vs Lesser vs Fuzzy
//!   counts.
//! - **Unaligned reason histogram** — why dropped extractions
//!   dropped.
//! - **Retry counts** — how often chunks had to retry.
//! - **Cache hits** — how many LLM calls were avoided.
//! - **Elapsed time** total and per chunk.
//!
//! # Health thresholds
//!
//! [`DocumentHealthThresholds`] defines "what does a healthy document
//! look like?" — minimum grounding rate, minimum extraction count,
//! maximum retry count, maximum elapsed time. A
//! [`DocumentReport`] self-scores against these and produces a
//! [`HealthStatus`] (`Healthy`, `Warning`, `Unhealthy`). The batch
//! runner uses the status to route documents to the right
//! downstream stage.

use std::collections::HashMap;
use std::time::Duration;

use langextract_aligner::UnalignedReason;
use langextract_core::AlignmentStatus;

/// Top-level document health status.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HealthStatus {
    /// Document meets every configured threshold. Safe to promote
    /// to downstream stages without human review.
    Healthy,
    /// Document passes minimum thresholds but has soft warning
    /// conditions (e.g. unusually many fuzzy matches). Promote
    /// cautiously.
    Warning,
    /// Document fails at least one minimum threshold. Should be
    /// routed to a manual-review queue before entering the rule IR.
    Unhealthy,
}

/// Per-chunk statistics captured during a single pipeline run.
#[derive(Debug, Clone)]
pub struct ChunkReport {
    /// Zero-based index of the chunk in the document.
    pub chunk_index: usize,
    /// Byte length of the chunk text.
    pub chunk_bytes: usize,
    /// Number of extractions the LLM returned for this chunk.
    pub emitted: usize,
    /// Number of extractions that got grounded (any status).
    pub grounded: usize,
    /// Number of retries the chunk needed (0 = first attempt
    /// succeeded).
    pub retries: u32,
    /// Whether the response came from the chunk cache.
    pub cache_hit: bool,
    /// Total elapsed time for this chunk, including retries and
    /// backoff.
    pub elapsed: Duration,
    /// Unaligned-reason counts for this chunk.
    pub unaligned_reasons: HashMap<&'static str, usize>,
}

/// Aggregate report for one document.
#[derive(Debug, Clone)]
pub struct DocumentReport {
    /// The document id (either user-supplied or auto-generated).
    pub document_id: String,
    /// Chunk-level breakdown.
    pub chunks: Vec<ChunkReport>,
    /// Total extractions emitted across all chunks.
    pub total_emitted: usize,
    /// Total grounded extractions.
    pub total_grounded: usize,
    /// Count by alignment status.
    pub alignment_counts: AlignmentCounts,
    /// Unaligned-reason histogram across the whole document.
    pub unaligned_reason_histogram: HashMap<&'static str, usize>,
    /// Total elapsed wall-clock time.
    pub elapsed: Duration,
    /// Total retry attempts across all chunks.
    pub total_retries: u32,
    /// Cache hit count.
    pub cache_hits: usize,
    /// Cache miss count.
    pub cache_misses: usize,
}

/// Breakdown of grounded extractions by [`AlignmentStatus`].
#[derive(Debug, Clone, Copy, Default)]
pub struct AlignmentCounts {
    /// `MatchExact` count.
    pub exact: usize,
    /// `MatchGreater` count (rare in practice).
    pub greater: usize,
    /// `MatchLesser` count.
    pub lesser: usize,
    /// `MatchFuzzy` count.
    pub fuzzy: usize,
}

impl AlignmentCounts {
    /// Tally an extraction's status. `None` statuses are skipped.
    #[expect(
        clippy::missing_const_for_fn,
        reason = "non-const for forward compat with counting strategies"
    )]
    pub fn add(&mut self, status: Option<AlignmentStatus>) {
        match status {
            Some(AlignmentStatus::MatchExact) => self.exact += 1,
            Some(AlignmentStatus::MatchGreater) => self.greater += 1,
            Some(AlignmentStatus::MatchLesser) => self.lesser += 1,
            Some(AlignmentStatus::MatchFuzzy) => self.fuzzy += 1,
            _ => {}
        }
    }

    /// Total count across all alignment statuses.
    #[must_use]
    pub const fn total(self) -> usize {
        self.exact + self.greater + self.lesser + self.fuzzy
    }
}

/// Thresholds that define a "healthy" document.
#[derive(Debug, Clone, Copy)]
pub struct DocumentHealthThresholds {
    /// Minimum grounding rate (0.0..=1.0). Documents below this are
    /// `Unhealthy`. Default 0.70.
    pub min_grounding_rate: f32,
    /// Warning threshold for grounding rate. Documents between
    /// `min_grounding_rate` and this value are `Warning`.
    /// Default 0.90.
    pub warning_grounding_rate: f32,
    /// Minimum number of extractions the document must produce to
    /// be considered healthy. Zero extractions on a non-empty
    /// document is almost always a prompt bug. Default 1.
    pub min_extractions: usize,
    /// Maximum total retries across the whole document before
    /// flagging. Default 20.
    pub max_total_retries: u32,
    /// Maximum fraction of extractions that can be `MatchFuzzy` or
    /// `MatchLesser` before flagging `Warning`. Default 0.25.
    pub max_fuzzy_fraction: f32,
}

impl Default for DocumentHealthThresholds {
    fn default() -> Self {
        Self {
            min_grounding_rate: 0.70,
            warning_grounding_rate: 0.90,
            min_extractions: 1,
            max_total_retries: 20,
            max_fuzzy_fraction: 0.25,
        }
    }
}

impl DocumentReport {
    /// Create an empty report for the given document id. Chunks
    /// are appended during the pipeline run.
    #[must_use]
    pub fn new(document_id: impl Into<String>) -> Self {
        Self {
            document_id: document_id.into(),
            chunks: Vec::new(),
            total_emitted: 0,
            total_grounded: 0,
            alignment_counts: AlignmentCounts::default(),
            unaligned_reason_histogram: HashMap::new(),
            elapsed: Duration::ZERO,
            total_retries: 0,
            cache_hits: 0,
            cache_misses: 0,
        }
    }

    /// Grounding rate as a `[0.0, 1.0]` fraction. Returns 1.0 for
    /// an empty document to avoid a division artefact.
    #[expect(
        clippy::cast_precision_loss,
        reason = "extraction counts are bounded well below f32 mantissa precision"
    )]
    #[must_use]
    pub fn grounding_rate(&self) -> f32 {
        if self.total_emitted == 0 {
            return 1.0;
        }
        self.total_grounded as f32 / self.total_emitted as f32
    }

    /// Fraction of grounded extractions that are fuzzy or lesser
    /// (as opposed to exact). High values indicate noisy
    /// extraction; the IR promotion pipeline should route these
    /// documents for manual review.
    #[expect(
        clippy::cast_precision_loss,
        reason = "extraction counts are bounded well below f32 mantissa precision"
    )]
    #[must_use]
    pub fn fuzzy_fraction(&self) -> f32 {
        let total = self.alignment_counts.total();
        if total == 0 {
            return 0.0;
        }
        (self.alignment_counts.fuzzy + self.alignment_counts.lesser) as f32 / total as f32
    }

    /// Score this report against the given thresholds.
    #[must_use]
    pub fn health(&self, thresholds: &DocumentHealthThresholds) -> HealthStatus {
        let gr = self.grounding_rate();
        if gr < thresholds.min_grounding_rate
            || self.total_emitted < thresholds.min_extractions
            || self.total_retries > thresholds.max_total_retries
        {
            return HealthStatus::Unhealthy;
        }
        if gr < thresholds.warning_grounding_rate
            || self.fuzzy_fraction() > thresholds.max_fuzzy_fraction
        {
            return HealthStatus::Warning;
        }
        HealthStatus::Healthy
    }

    /// Number of chunks processed.
    #[must_use]
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }
}

/// Convert a [`UnalignedReason`] to the stable discriminant string.
///
/// Used by the report's histogram. Kept here so downstream consumers
/// don't depend directly on the aligner's reason type for their
/// dashboards.
#[must_use]
#[expect(
    clippy::missing_const_for_fn,
    reason = "non-const for forward compat with cross-crate strings"
)]
pub fn unaligned_reason_label(reason: &UnalignedReason) -> &'static str {
    match reason {
        UnalignedReason::NoExactMatch => "NoExactMatch",
        UnalignedReason::LesserMatchRejected { .. } => "LesserMatchRejected",
        UnalignedReason::BelowFuzzyThreshold { .. } => "BelowFuzzyThreshold",
        UnalignedReason::FuzzyDisabled => "FuzzyDisabled",
        UnalignedReason::SkippedBySafeguard { .. } => "SkippedBySafeguard",
        UnalignedReason::EmptyExtractionTokens => "EmptyExtractionTokens",
        // UnalignedReason is non_exhaustive — future variants land here.
        _ => "Other",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn health_healthy() {
        let mut r = DocumentReport::new("d1");
        r.total_emitted = 10;
        r.total_grounded = 10;
        r.alignment_counts.exact = 10;
        assert_eq!(
            r.health(&DocumentHealthThresholds::default()),
            HealthStatus::Healthy
        );
    }

    #[test]
    fn health_warning_on_low_grounding() {
        let mut r = DocumentReport::new("d1");
        r.total_emitted = 10;
        r.total_grounded = 8;
        r.alignment_counts.exact = 8;
        assert_eq!(
            r.health(&DocumentHealthThresholds::default()),
            HealthStatus::Warning
        );
    }

    #[test]
    fn health_unhealthy_on_very_low_grounding() {
        let mut r = DocumentReport::new("d1");
        r.total_emitted = 10;
        r.total_grounded = 5;
        r.alignment_counts.exact = 5;
        assert_eq!(
            r.health(&DocumentHealthThresholds::default()),
            HealthStatus::Unhealthy
        );
    }

    #[test]
    fn health_warning_on_fuzzy_heavy() {
        let mut r = DocumentReport::new("d1");
        r.total_emitted = 10;
        r.total_grounded = 10;
        r.alignment_counts.fuzzy = 5;
        r.alignment_counts.exact = 5;
        assert_eq!(
            r.health(&DocumentHealthThresholds::default()),
            HealthStatus::Warning
        );
    }
}
