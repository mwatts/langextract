//! Sliding-window fuzzy alignment of a single extraction against the
//! source tokens.
//!
//! Port of `_fuzzy_align_extraction` in `langextract/resolver.py`.
//!
//! The algorithm tries every window size from
//! `len(extraction_tokens)..=len(source_tokens)` and every position
//! within each window size. For each window it computes:
//!
//! 1. A **Counter-intersection upper bound** — the sum of
//!    `min(count_in_window, count_in_extraction)` across the normalized
//!    token multiset. If that upper bound is below
//!    `len_e * threshold`, the window cannot possibly meet the ratio
//!    threshold and is skipped without running the full diff.
//! 2. A full diff (via the `similar` crate) comparing the window's
//!    normalized tokens against the extraction's normalized tokens. The
//!    ratio is `matches / len(extraction_tokens)`.
//!
//! The best window whose ratio is ≥ threshold wins, yielding a
//! [`TokenInterval`] / [`CharInterval`] pair and a
//! [`AlignmentStatus::MatchFuzzy`] caller-side.
//!
//! Normalization — lowercase plus trailing-`s` stemming — lets
//! `"problems"` match `"problem"` and similar minor variations.

use std::collections::HashMap;

use langextract_core::{CharInterval, Extraction, TokenInterval};
use langextract_tokenizer::{Token, Tokenizer};
use similar::DiffOp;

use crate::align::{char_interval_for_span, diff_token_slices};
use crate::normalize::{lowercase_tokens_from, normalize_tokens};

/// Try to align a single extraction to the source via sliding-window
/// fuzzy matching. Returns the aligned `(TokenInterval, CharInterval)`
/// pair if a window meets the threshold, or `None` if not.
pub(crate) fn fuzzy_align<T: Tokenizer>(
    extraction: &Extraction,
    source_tokens_lower: &[String],
    source_tokens_raw: &[Token],
    threshold: f32,
    token_offset: usize,
    char_offset: usize,
    tokenizer: &T,
) -> Option<(TokenInterval, CharInterval)> {
    let extraction_lower =
        lowercase_tokens_from(&tokenizer.tokenize(&extraction.extraction_text));
    if extraction_lower.is_empty() {
        return None;
    }
    let extraction_norm = normalize_tokens(&extraction_lower);
    let len_e = extraction_norm.len();

    // Pre-compute multiset counts for the whole extraction.
    let extraction_counts = counts(&extraction_norm);

    // Python: `int(len_e * fuzzy_alignment_threshold)` — truncating.
    // `len_e` is a token count (realistically < 10_000), so the
    // round-trip through f32 cannot lose precision in practice.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "token counts are bounded well below f32 mantissa precision"
    )]
    let min_overlap: usize = (len_e as f32 * threshold) as usize;

    let mut best_ratio: f32 = 0.0;
    let mut best_span: Option<(usize, usize)> = None; // (start, window_size)

    let max_window = source_tokens_lower.len();
    if max_window < len_e {
        return None;
    }

    for window_size in len_e..=max_window {
        // Initial window counts.
        let initial_window_norm: Vec<String> =
            normalize_tokens(&source_tokens_lower[..window_size]);
        let mut window_counts = counts(&initial_window_norm);

        // Buffer of normalized tokens in the window, for sliding.
        let mut window_norm: std::collections::VecDeque<String> =
            initial_window_norm.into();

        let max_start = source_tokens_lower.len() - window_size;
        for start_idx in 0..=max_start {
            // Counter-intersection upper bound.
            if counter_intersection_total(&extraction_counts, &window_counts) >= min_overlap {
                // Full diff to compute exact matches.
                let window_slice: Vec<&String> = window_norm.iter().collect();
                let matches = matching_token_count(&window_slice, &extraction_norm);
                #[expect(
                    clippy::cast_precision_loss,
                    reason = "token counts are bounded well below f32 mantissa precision"
                )]
                let ratio = if len_e > 0 {
                    matches as f32 / len_e as f32
                } else {
                    0.0
                };
                if ratio > best_ratio {
                    best_ratio = ratio;
                    best_span = Some((start_idx, window_size));
                }
            }

            // Slide the window to the right by one.
            if start_idx + window_size < source_tokens_lower.len() {
                let old = window_norm.pop_front().expect("non-empty window");
                decrement(&mut window_counts, &old);
                let next_idx = start_idx + window_size;
                let new_tok =
                    crate::normalize::normalize_token(&source_tokens_lower[next_idx]);
                increment(&mut window_counts, new_tok.clone());
                window_norm.push_back(new_tok);
            }
        }
    }

    let (start_idx, window_size) = best_span?;
    if best_ratio < threshold {
        return None;
    }

    let token_interval = TokenInterval::new(
        start_idx + token_offset,
        start_idx + window_size + token_offset,
    );
    let char_interval =
        char_interval_for_span(source_tokens_raw, start_idx, window_size, char_offset);
    Some((token_interval, char_interval))
}

// ---------- multiset helpers ----------

fn counts(tokens: &[String]) -> HashMap<String, usize> {
    let mut c = HashMap::with_capacity(tokens.len());
    for t in tokens {
        *c.entry(t.clone()).or_insert(0) += 1;
    }
    c
}

fn increment(c: &mut HashMap<String, usize>, key: String) {
    *c.entry(key).or_insert(0) += 1;
}

fn decrement(c: &mut HashMap<String, usize>, key: &str) {
    if let Some(n) = c.get_mut(key) {
        *n -= 1;
        if *n == 0 {
            c.remove(key);
        }
    }
}

/// Total size of the multiset intersection of two counter maps.
/// `Σ min(a[k], b[k])` — this is the upper bound on how many tokens of
/// `a` could possibly match tokens of `b` regardless of order.
fn counter_intersection_total(
    a: &HashMap<String, usize>,
    b: &HashMap<String, usize>,
) -> usize {
    // Iterate the smaller of the two maps for efficiency.
    let (small, large) = if a.len() <= b.len() { (a, b) } else { (b, a) };
    small
        .iter()
        .map(|(k, &v)| v.min(*large.get(k).unwrap_or(&0)))
        .sum()
}

/// Count matching tokens between two slices using the `similar` diff
/// algorithm — the Rust analogue of `sum(size for _, _, size in
/// matcher.get_matching_blocks())`.
fn matching_token_count(a: &[&String], b: &[String]) -> usize {
    // Copy to owned `String` slices so both sides have the same type
    // for the diff. Cheap because the fuzzy phase runs only on the
    // (usually small) set of extractions that failed exact matching.
    let a_owned: Vec<String> = a.iter().map(|s| (*s).clone()).collect();
    diff_token_slices(&a_owned, b)
        .into_iter()
        .filter_map(|op| match op {
            DiffOp::Equal { len, .. } => Some(len),
            _ => None,
        })
        .sum()
}
