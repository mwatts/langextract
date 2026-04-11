//! Diagnostics explaining *why* an extraction failed to align.
//!
//! The base [`align_extraction_groups`](crate::align_extraction_groups)
//! API returns grounded extractions and silently leaves unaligned ones
//! with `char_interval == None`, `alignment_status == None`. That's
//! fine for simple callers, but at scale — especially when debugging
//! prompts, measuring extraction quality, or promoting results to a
//! downstream rule-extraction layer — you need to know *why* each
//! dropped extraction dropped.
//!
//! [`align_extraction_groups_with_diagnostics`](crate::align_extraction_groups_with_diagnostics)
//! returns an [`AlignmentReport`] that pairs the grounded groups with
//! a map of [`UnalignedReason`]s keyed on the extraction's
//! `(group_index, ext_index_in_group)` coordinate.

use std::collections::HashMap;

use langextract_core::Extraction;

/// Why a single extraction failed to align to the source.
///
/// Ordered roughly from "most likely a prompt bug" at the top to
/// "most likely a safeguard firing" at the bottom.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum UnalignedReason {
    /// The exact-match phase's diff did not find any block whose
    /// new-side index landed at the start of this extraction's token
    /// range. The extraction either shares no tokens with the
    /// source or the tokens that do match happened to be grabbed by
    /// an earlier extraction in the delimited stream. Usually means
    /// the LLM hallucinated the extraction text.
    NoExactMatch,

    /// The exact-match phase found a partial block but the
    /// extraction was longer than the block, and
    /// [`AlignmentOptions::accept_match_lesser`](crate::AlignmentOptions::accept_match_lesser)
    /// is `false`. The fuzzy phase then either found no improvement
    /// or was disabled. Useful distinction because
    /// `accept_match_lesser` is a tuning knob, not a prompt bug.
    LesserMatchRejected {
        /// Token count of the block the exact phase found.
        matched_len: usize,
        /// Token count of the whole extraction.
        extraction_len: usize,
    },

    /// Fuzzy phase ran but no window scored ≥ the configured
    /// threshold. Carries the best ratio observed so callers can
    /// see how close it got — useful for tuning the threshold.
    BelowFuzzyThreshold {
        /// Best ratio observed across all candidate windows.
        best_ratio: f32,
        /// The threshold that was not met.
        threshold: f32,
    },

    /// Fuzzy phase was disabled in
    /// [`AlignmentOptions`](crate::AlignmentOptions) and the exact
    /// phase produced no qualifying match.
    FuzzyDisabled,

    /// Fuzzy phase was skipped for this extraction because a
    /// [`FuzzySafeguards`](crate::FuzzySafeguards) limit fired —
    /// typically the per-chunk extraction cap.
    SkippedBySafeguard {
        /// Human-readable reason ("too many extractions in chunk",
        /// etc.). Stable enough to match on in tests but not an
        /// enum variant because the set of safeguards is expected
        /// to grow.
        reason: &'static str,
    },

    /// The extraction's own tokenization was empty (e.g. the LLM
    /// emitted an all-whitespace extraction text).
    EmptyExtractionTokens,
}

/// The result of
/// [`align_extraction_groups_with_diagnostics`](crate::align_extraction_groups_with_diagnostics):
/// the groups with their grounding filled in, plus a map of unaligned
/// reasons.
#[derive(Debug, Clone)]
pub struct AlignmentReport {
    /// The extraction groups, with `char_interval`, `token_interval`,
    /// and `alignment_status` populated on successful alignments.
    /// Unaligned extractions are left untouched (all three fields
    /// remain `None`).
    pub groups: Vec<Vec<Extraction>>,

    /// Why each unaligned extraction dropped, keyed on
    /// `(group_index, extraction_index_in_group)`.
    pub unaligned_reasons: HashMap<(usize, usize), UnalignedReason>,
}

impl AlignmentReport {
    /// Count of grounded extractions across all groups.
    #[must_use]
    pub fn grounded_count(&self) -> usize {
        self.groups
            .iter()
            .flatten()
            .filter(|e| e.char_interval.is_some())
            .count()
    }

    /// Count of unaligned extractions.
    #[must_use]
    pub fn unaligned_count(&self) -> usize {
        self.unaligned_reasons.len()
    }

    /// Total extraction count (aligned + unaligned).
    #[must_use]
    pub fn total_count(&self) -> usize {
        self.groups.iter().map(Vec::len).sum()
    }

    /// Grounding rate as a fraction in `[0.0, 1.0]`. Returns `1.0`
    /// for an empty report to avoid a division artefact in
    /// downstream reporting.
    #[must_use]
    #[expect(
        clippy::cast_precision_loss,
        reason = "extraction counts are bounded well below f32 mantissa precision"
    )]
    pub fn grounding_rate(&self) -> f32 {
        let total = self.total_count();
        if total == 0 {
            return 1.0;
        }
        self.grounded_count() as f32 / total as f32
    }

    /// Histogram of unaligned reasons by discriminant. Useful for
    /// dashboarding: "40% `NoExactMatch`, 30% `BelowFuzzyThreshold`, …".
    #[must_use]
    pub fn reason_histogram(&self) -> HashMap<&'static str, usize> {
        let mut h: HashMap<&'static str, usize> = HashMap::new();
        for reason in self.unaligned_reasons.values() {
            let key = match reason {
                UnalignedReason::NoExactMatch => "NoExactMatch",
                UnalignedReason::LesserMatchRejected { .. } => "LesserMatchRejected",
                UnalignedReason::BelowFuzzyThreshold { .. } => "BelowFuzzyThreshold",
                UnalignedReason::FuzzyDisabled => "FuzzyDisabled",
                UnalignedReason::SkippedBySafeguard { .. } => "SkippedBySafeguard",
                UnalignedReason::EmptyExtractionTokens => "EmptyExtractionTokens",
            };
            *h.entry(key).or_insert(0) += 1;
        }
        h
    }
}
