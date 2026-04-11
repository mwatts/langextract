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

use crate::diagnostics::{AlignmentReport, UnalignedReason};
use crate::error::AlignError;
use crate::fuzzy::{FuzzyOutcome, fuzzy_align};
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

/// Safeguards for the fuzzy alignment phase.
///
/// Fuzzy alignment is `O(extractions × windows × diff_cost)`, which is
/// fine for normal LLM output (~10-20 careful extractions per chunk,
/// most exact-matching) but can blow up on misbehaving or noisy model
/// responses. Every field here is a pressure-release valve the
/// pipeline can cap to keep a single bad response from taking down a
/// whole batch.
#[derive(Debug, Clone, Copy)]
pub struct FuzzySafeguards {
    /// Maximum number of extractions per chunk that may enter the
    /// fuzzy phase. Anything beyond this gets marked
    /// [`UnalignedReason::TooManyExtractions`] and skipped. A real
    /// LLM rarely emits more than ~20 per chunk; setting this to
    /// 100 catches pathological responses without rejecting
    /// legitimate high-density output.
    pub max_fuzzy_extractions_per_chunk: usize,

    /// Maximum window size to try during the sliding-window scan,
    /// as a multiplier of the extraction's own token length. A
    /// value of 3.0 means "try windows up to 3× the extraction
    /// length". The Python implementation tries every window up to
    /// the full source length, which is quadratic in source size.
    pub max_window_size_multiplier: f32,

    /// Hard upper bound on window size, regardless of multiplier.
    /// Prevents the multiplier from degenerating on long chunks.
    pub max_window_size_absolute: usize,
}

impl Default for FuzzySafeguards {
    fn default() -> Self {
        Self {
            max_fuzzy_extractions_per_chunk: 100,
            max_window_size_multiplier: 3.0,
            max_window_size_absolute: 64,
        }
    }
}

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

    /// Safety limits on the fuzzy alignment phase. See
    /// [`FuzzySafeguards`].
    pub fuzzy_safeguards: FuzzySafeguards,
}

impl Default for AlignmentOptions {
    fn default() -> Self {
        Self {
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: DEFAULT_FUZZY_THRESHOLD,
            accept_match_lesser: true,
            token_offset: 0,
            char_offset: 0,
            fuzzy_safeguards: FuzzySafeguards::default(),
        }
    }
}

/// Align extractions with the default [`RegexTokenizer`].
///
/// Returns just the grounded groups; unaligned extractions are
/// silently left with `char_interval == None`. See
/// [`align_extraction_groups_with_diagnostics`] if you want the
/// reasons.
pub fn align_extraction_groups(
    groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
) -> Result<Vec<Vec<Extraction>>, AlignError> {
    align_extraction_groups_with(groups, source_text, options, &RegexTokenizer::new())
}

/// Align extractions with diagnostics, using the default
/// [`RegexTokenizer`]. Returns an [`AlignmentReport`] that pairs the
/// grounded groups with a map of [`UnalignedReason`]s.
pub fn align_extraction_groups_with_diagnostics(
    groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
) -> Result<AlignmentReport, AlignError> {
    align_extraction_groups_with_diagnostics_and(
        groups,
        source_text,
        options,
        &RegexTokenizer::new(),
    )
}

/// Align extractions using a custom tokenizer.
///
/// The tokenizer must be the same one used to build any token indices
/// you pass back into the pipeline later, and should handle the
/// delimiter ([`DEFAULT_DELIMITER`]) as a single token.
pub fn align_extraction_groups_with<T: Tokenizer>(
    groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
    tokenizer: &T,
) -> Result<Vec<Vec<Extraction>>, AlignError> {
    let report = align_extraction_groups_with_diagnostics_and(
        groups, source_text, options, tokenizer,
    )?;
    Ok(report.groups)
}

/// Per-extraction state tracked during the exact-match phase so the
/// fuzzy phase can pick up where the exact phase left off and
/// diagnostics can record the precise reason each failed.
#[derive(Clone, Copy)]
enum ExactState {
    Aligned,
    LesserRejected { matched: usize, total: usize },
    NotFound,
}

/// Align extractions with diagnostics, using a custom tokenizer.
#[expect(
    clippy::too_many_lines,
    reason = "verbatim composition of the exact + fuzzy phases with their \
              diagnostics bookkeeping; splitting would obscure the \
              per-extraction state machine"
)]
pub fn align_extraction_groups_with_diagnostics_and<T: Tokenizer>(
    mut groups: Vec<Vec<Extraction>>,
    source_text: &str,
    options: &AlignmentOptions,
    tokenizer: &T,
) -> Result<AlignmentReport, AlignError> {
    if groups.is_empty() {
        return Ok(AlignmentReport {
            groups,
            unaligned_reasons: HashMap::new(),
        });
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
    let mut state_map: HashMap<(usize, usize), ExactState> = HashMap::new();
    for (group_idx, group) in groups.iter().enumerate() {
        for ext_idx in 0..group.len() {
            state_map.insert((group_idx, ext_idx), ExactState::NotFound);
        }
    }

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
            continue;
        };

        let extraction = &mut groups[g][e];
        let ext_len = ext_lengths[&(g, e)];

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
            state_map.insert((g, e), ExactState::Aligned);
        } else if options.accept_match_lesser {
            extraction.token_interval = Some(token_interval);
            extraction.char_interval = Some(char_interval);
            extraction.alignment_status = Some(AlignmentStatus::MatchLesser);
            state_map.insert((g, e), ExactState::Aligned);
        } else {
            state_map.insert(
                (g, e),
                ExactState::LesserRejected {
                    matched: len,
                    total: ext_len,
                },
            );
        }
    }

    // Collect per-extraction state into `unaligned_reasons`. Start
    // by recording the exact-phase reason; fuzzy phase may
    // overwrite with a better reason (or promote to aligned).
    let mut unaligned_reasons: HashMap<(usize, usize), UnalignedReason> = HashMap::new();
    let mut fuzzy_candidates: Vec<(usize, usize)> = Vec::new();
    for ((g, e), state) in &state_map {
        match state {
            ExactState::Aligned => { /* grounded, nothing to record */ }
            ExactState::NotFound => {
                unaligned_reasons.insert((*g, *e), UnalignedReason::NoExactMatch);
                fuzzy_candidates.push((*g, *e));
            }
            ExactState::LesserRejected { matched, total } => {
                unaligned_reasons.insert(
                    (*g, *e),
                    UnalignedReason::LesserMatchRejected {
                        matched_len: *matched,
                        extraction_len: *total,
                    },
                );
                fuzzy_candidates.push((*g, *e));
            }
        }
    }

    // Fuzzy phase.
    if options.enable_fuzzy_alignment {
        // Safeguard: cap the number of extractions entering fuzzy.
        let safeguards = options.fuzzy_safeguards;
        let cap = safeguards.max_fuzzy_extractions_per_chunk;
        let (to_run, to_skip) = if fuzzy_candidates.len() > cap {
            let split = cap;
            (&fuzzy_candidates[..split], &fuzzy_candidates[split..])
        } else {
            (&fuzzy_candidates[..], &[][..])
        };
        for (g, e) in to_skip {
            unaligned_reasons.insert(
                (*g, *e),
                UnalignedReason::SkippedBySafeguard {
                    reason: "max_fuzzy_extractions_per_chunk exceeded",
                },
            );
        }

        for (g, e) in to_run {
            let extraction = &mut groups[*g][*e];
            let outcome = fuzzy_align(
                extraction,
                &source_tokens,
                &source_tokenized.tokens,
                options.fuzzy_alignment_threshold,
                options.token_offset,
                options.char_offset,
                &safeguards,
                tokenizer,
            );
            match outcome {
                FuzzyOutcome::Aligned {
                    token_interval,
                    char_interval,
                    ratio: _,
                } => {
                    extraction.token_interval = Some(token_interval);
                    extraction.char_interval = Some(char_interval);
                    extraction.alignment_status = Some(AlignmentStatus::MatchFuzzy);
                    unaligned_reasons.remove(&(*g, *e));
                }
                FuzzyOutcome::EmptyExtraction => {
                    unaligned_reasons
                        .insert((*g, *e), UnalignedReason::EmptyExtractionTokens);
                }
                FuzzyOutcome::SourceShorterThanExtraction => {
                    // Leave whatever reason we set in the exact
                    // phase — NoExactMatch or LesserMatchRejected.
                }
                FuzzyOutcome::BelowThreshold { best_ratio } => {
                    unaligned_reasons.insert(
                        (*g, *e),
                        UnalignedReason::BelowFuzzyThreshold {
                            best_ratio,
                            threshold: options.fuzzy_alignment_threshold,
                        },
                    );
                }
            }
        }
    } else {
        // Fuzzy disabled: replace NoExactMatch reasons with
        // FuzzyDisabled for clarity. LesserMatchRejected stays
        // as-is because it's more specific.
        for (g, e) in &fuzzy_candidates {
            if matches!(
                unaligned_reasons.get(&(*g, *e)),
                Some(UnalignedReason::NoExactMatch)
            ) {
                unaligned_reasons.insert((*g, *e), UnalignedReason::FuzzyDisabled);
            }
        }
    }

    Ok(AlignmentReport {
        groups,
        unaligned_reasons,
    })
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
