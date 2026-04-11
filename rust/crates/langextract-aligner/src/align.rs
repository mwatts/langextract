//! Token-based alignment of LLM extractions against source text.
//!
//! Port of `WordAligner` from `langextract/resolver.py`. The algorithm
//! runs in two phases:
//!
//! 1. **Exact match phase.** All extractions are concatenated with a
//!    delimiter token and matched against the source via a
//!    [`similar`]-powered `SequenceMatcher`-equivalent. Each matching
//!    block whose start index coincides with the start of some
//!    extraction's token range is consumed to align that extraction.
//!    - Full coverage → [`AlignmentStatus::MatchExact`].
//!    - Partial coverage (extraction has more tokens than the block) →
//!      [`AlignmentStatus::MatchLesser`], unless
//!      [`AlignmentOptions::accept_match_lesser`] is `false`, in which
//!      case the extraction is left unaligned for the fuzzy phase.
//!
//! 2. **Fuzzy phase.** Any extraction left unaligned is scanned
//!    per-extraction with a sliding window over `source_tokens`. A
//!    Counter-based upper bound skips obviously-bad windows. The best
//!    window whose matching ratio is ≥
//!    [`AlignmentOptions::fuzzy_alignment_threshold`] becomes a
//!    [`AlignmentStatus::MatchFuzzy`] result.
//!
//! This module **does not** parse LLM output into extractions — that's
//! the resolver/parser layer's job, deferred until
//! `format_handler.py` is ported. The caller supplies already-parsed
//! [`Extraction`]s grouped into [`Vec<Vec<Extraction>>`] and gets them
//! back with `token_interval`, `char_interval`, and `alignment_status`
//! populated where alignment succeeded.

use std::collections::HashMap;

use langextract_core::{AlignmentStatus, CharInterval, Extraction, TokenInterval};
use langextract_tokenizer::{RegexTokenizer, Token, Tokenizer};
use similar::DiffOp;
use similar::algorithms::{Algorithm, Capture, diff_slices};

use crate::error::AlignError;
use crate::fuzzy::fuzzy_align;
use crate::normalize::lowercase_tokens_from;

/// The default minimum ratio a fuzzy match must hit to be accepted.
/// Mirrors `_FUZZY_ALIGNMENT_MIN_THRESHOLD` in `resolver.py`.
pub const DEFAULT_FUZZY_THRESHOLD: f32 = 0.75;

/// Default delimiter used to join extraction texts during exact matching.
///
/// The Unicode unit separator (U+241F) is chosen because it tokenizes to
/// a single punctuation token under the default [`RegexTokenizer`] and
/// is virtually never present in LLM output.
pub const DEFAULT_DELIMITER: &str = "\u{241F}";

/// Options controlling alignment behaviour.
#[derive(Debug, Clone)]
pub struct AlignmentOptions {
    /// Whether to fall back to per-extraction sliding-window fuzzy
    /// matching when the exact phase fails.
    pub enable_fuzzy_alignment: bool,

    /// Minimum ratio of matched tokens to extraction tokens required
    /// for a fuzzy match to be accepted. Range 0.0..=1.0.
    pub fuzzy_alignment_threshold: f32,

    /// Whether to accept partial exact matches (extraction longer than
    /// the matched block) as [`AlignmentStatus::MatchLesser`]. If
    /// `false`, such extractions are left for the fuzzy phase.
    pub accept_match_lesser: bool,

    /// Token offset to add to the start of each computed token
    /// interval (used when aligning a chunk of a larger document).
    pub token_offset: usize,

    /// Character offset to add to the start of each computed character
    /// interval (used when aligning a chunk of a larger document).
    pub char_offset: usize,
}

impl Default for AlignmentOptions {
    fn default() -> Self {
        Self {
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: DEFAULT_FUZZY_THRESHOLD,
            accept_match_lesser: true,
            token_offset: 0,
            char_offset: 0,
        }
    }
}

/// Align extractions with the default [`RegexTokenizer`].
pub fn align_extraction_groups(
    groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
) -> Result<Vec<Vec<Extraction>>, AlignError> {
    align_extraction_groups_with(groups, source_text, options, &RegexTokenizer::new())
}

/// Align extractions using a custom tokenizer.
///
/// The tokenizer must be the same one used to build any token indices
/// you pass back into the pipeline later, and should handle the
/// delimiter ([`DEFAULT_DELIMITER`]) as a single token.
pub fn align_extraction_groups_with<T: Tokenizer>(
    mut groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
    tokenizer: &T,
) -> Result<Vec<Vec<Extraction>>, AlignError> {
    if groups.is_empty() {
        return Ok(groups);
    }

    // Tokenize source once. We need both the `TokenizedText` (for
    // char-interval lookups) and the lowercase token strings (for
    // matching).
    let source_tokenized = tokenizer.tokenize(source_text);
    let source_tokens = lowercase_tokens_from(&source_tokenized);

    // Validate the delimiter tokenizes to exactly one token.
    let delim_tokens = lowercase_tokens_from(&tokenizer.tokenize(DEFAULT_DELIMITER));
    if delim_tokens.len() != 1 {
        return Err(AlignError::InvalidDelimiter {
            delimiter: DEFAULT_DELIMITER.to_owned(),
            count: delim_tokens.len(),
        });
    }

    // Validate no extraction contains the delimiter in its raw text.
    for group in &groups {
        for ex in group {
            if ex.extraction_text.contains(DEFAULT_DELIMITER) {
                return Err(AlignError::DelimiterInExtraction {
                    delimiter: DEFAULT_DELIMITER.to_owned(),
                    extraction_text: ex.extraction_text.clone(),
                });
            }
        }
    }

    // Build the joined extraction tokens and a map from "j-start" (the
    // starting index of an extraction's tokens inside the joined
    // stream) to its (group_idx, ext_idx_in_group) coordinate.
    let joined = join_with_delimiter(&groups, &format!(" {DEFAULT_DELIMITER} "));
    let extraction_tokens = lowercase_tokens_from(&tokenizer.tokenize(&joined));

    let (index_map, ext_lengths) = build_extraction_index_map(&groups, tokenizer);

    // Diff source_tokens ↔ extraction_tokens. similar's DiffOp::Equal
    // has the same (old_index, new_index, len) shape as difflib's
    // (i, j, n).
    let ops = diff_token_slices(&source_tokens, &extraction_tokens);

    // Exact phase.
    let mut aligned_flat_keys: Vec<(usize, usize)> = Vec::new();
    for op in ops {
        let DiffOp::Equal {
            old_index,
            new_index,
            len,
        } = op
        else {
            continue;
        };
        let Some(&(g, e)) = index_map.get(&new_index) else {
            // This block doesn't start at the beginning of any
            // extraction's token range — skip (matches Python's
            // "no clean start index found" branch).
            continue;
        };

        let extraction = &mut groups[g][e];
        let ext_len = ext_lengths[&(g, e)];

        // The block cannot exceed the extraction's own token count;
        // the delimiter between extractions guarantees it.
        debug_assert!(
            len <= ext_len,
            "diff block length {len} exceeds extraction token count {ext_len}"
        );

        let token_interval = TokenInterval::new(
            old_index + options.token_offset,
            old_index + len + options.token_offset,
        );
        let char_interval = char_interval_for_span(
            &source_tokenized.tokens,
            old_index,
            len,
            options.char_offset,
        );

        if ext_len == len {
            extraction.token_interval = Some(token_interval);
            extraction.char_interval = Some(char_interval);
            extraction.alignment_status = Some(AlignmentStatus::MatchExact);
            aligned_flat_keys.push((g, e));
        } else if options.accept_match_lesser {
            extraction.token_interval = Some(token_interval);
            extraction.char_interval = Some(char_interval);
            extraction.alignment_status = Some(AlignmentStatus::MatchLesser);
            aligned_flat_keys.push((g, e));
        }
        // Else: leave unaligned, fall through to fuzzy phase.
    }

    // Fuzzy phase.
    if options.enable_fuzzy_alignment {
        for (g_idx, group) in groups.iter_mut().enumerate() {
            for (e_idx, extraction) in group.iter_mut().enumerate() {
                if aligned_flat_keys.contains(&(g_idx, e_idx)) {
                    continue;
                }
                if let Some(aligned) = fuzzy_align(
                    extraction,
                    &source_tokens,
                    &source_tokenized.tokens,
                    options.fuzzy_alignment_threshold,
                    options.token_offset,
                    options.char_offset,
                    tokenizer,
                ) {
                    let (token_interval, char_interval) = aligned;
                    extraction.token_interval = Some(token_interval);
                    extraction.char_interval = Some(char_interval);
                    extraction.alignment_status = Some(AlignmentStatus::MatchFuzzy);
                }
            }
        }
    }

    Ok(groups)
}

// ---------- helpers ----------

/// Concatenate all extraction texts with a surrounding-space delimiter
/// so the matcher sees a clean single-token boundary between them.
fn join_with_delimiter(groups: &[Vec<Extraction>], delim_with_spaces: &str) -> String {
    let total_estimate: usize = groups
        .iter()
        .flat_map(|g| g.iter().map(|e| e.extraction_text.len()))
        .sum::<usize>()
        + groups.iter().map(Vec::len).sum::<usize>() * delim_with_spaces.len();
    let mut out = String::with_capacity(total_estimate);
    let mut first = true;
    for group in groups {
        for ex in group {
            if !first {
                out.push_str(delim_with_spaces);
            }
            out.push_str(&ex.extraction_text);
            first = false;
        }
    }
    out
}

/// Map from "j-start" (starting index of an extraction's token range in
/// the joined extraction stream) to its `(group_idx, ext_idx_in_group)`
/// coordinate.
type ExtractionStartMap = HashMap<usize, (usize, usize)>;

/// Map from `(group_idx, ext_idx_in_group)` to the extraction's own
/// lowercase-token length, excluding the delimiter.
type ExtractionLengthMap = HashMap<(usize, usize), usize>;

/// The delimiter between consecutive extractions occupies exactly one
/// token in the joined stream (validated by `InvalidDelimiter` above).
const DELIM_LEN: usize = 1;

/// Build the [`ExtractionStartMap`] and [`ExtractionLengthMap`] for a
/// group of extractions.
fn build_extraction_index_map<T: Tokenizer>(
    groups: &[Vec<Extraction>],
    tokenizer: &T,
) -> (ExtractionStartMap, ExtractionLengthMap) {
    let mut index_map: ExtractionStartMap = HashMap::new();
    let mut lengths: ExtractionLengthMap = HashMap::new();
    let mut cursor: usize = 0;
    for (g, group) in groups.iter().enumerate() {
        for (e, ex) in group.iter().enumerate() {
            let ext_tokens = lowercase_tokens_from(&tokenizer.tokenize(&ex.extraction_text));
            index_map.insert(cursor, (g, e));
            lengths.insert((g, e), ext_tokens.len());
            cursor += ext_tokens.len() + DELIM_LEN;
        }
    }
    (index_map, lengths)
}

/// Compute a [`CharInterval`] covering `[start_token_idx, start_token_idx + len)`
/// in a slice of source tokens, adding the chunk's `char_offset`.
pub(crate) fn char_interval_for_span(
    source_tokens: &[Token],
    start_token_idx: usize,
    len: usize,
    char_offset: usize,
) -> CharInterval {
    debug_assert!(len > 0, "char_interval_for_span needs len > 0");
    let start = source_tokens[start_token_idx].char_interval.start;
    let end = source_tokens[start_token_idx + len - 1].char_interval.end;
    CharInterval::new(char_offset + start, char_offset + end)
}

/// Thin wrapper around `similar`'s `diff_slices` + `Capture` so that
/// both the exact and fuzzy phases share a single "give me the matching
/// blocks between these two token slices" API.
pub(crate) fn diff_token_slices<T: Eq + std::hash::Hash + Ord>(
    old: &[T],
    new: &[T],
) -> Vec<DiffOp> {
    let mut capture = Capture::new();
    // `diff_slices` returns `Result<(), D::Error>` — for `Capture`,
    // `D::Error = Infallible`, so unwrap is guaranteed not to panic.
    diff_slices(Algorithm::Myers, &mut capture, old, new).expect("Capture is Infallible");
    capture.into_ops()
}
