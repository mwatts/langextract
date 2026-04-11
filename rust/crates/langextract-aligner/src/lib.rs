//! Token-based exact and fuzzy alignment of LLM extractions against
//! source text.
//!
//! This is the Rust port of the `WordAligner` class in
//! `langextract/resolver.py`. The parsing half of `resolver.py`
//! (JSON/YAML → `Vec<Extraction>`) depends on the not-yet-ported
//! `format_handler.py` and lives in a separate crate. This crate only
//! does alignment — it takes already-parsed [`Extraction`]s and
//! populates their `token_interval`, `char_interval`, and
//! `alignment_status` fields.
//!
//! # Algorithm
//!
//! Two phases:
//!
//! 1. **Exact phase** — all extractions are concatenated with a single
//!    delimiter token between them and matched against the source via
//!    a `SequenceMatcher`-equivalent from the [`similar`] crate.
//!    Matching blocks whose position lines up with the start of an
//!    extraction's token range consume that extraction. Full coverage
//!    → [`AlignmentStatus::MatchExact`]; partial coverage (extraction
//!    has more tokens than the block) → [`AlignmentStatus::MatchLesser`],
//!    unless that status is opted out of.
//!
//! 2. **Fuzzy phase** — any extraction left unaligned gets a per-item
//!    sliding-window scan over the source. Normalized token counts
//!    give a cheap upper bound on match quality, and only windows that
//!    could conceivably meet the ratio threshold get diffed. The best
//!    qualifying window becomes a [`AlignmentStatus::MatchFuzzy`] result.
//!
//! # What this is **not**
//!
//! - **Not a parser.** You supply [`Extraction`]s built from the LLM's
//!   output; this crate does not parse JSON or YAML.
//! - **Not a tokenizer.** Tokenization is delegated to
//!   [`langextract_tokenizer::Tokenizer`] implementations. The default
//!   path uses [`langextract_tokenizer::RegexTokenizer`].
//! - **Not pixel-identical to Python.** Python's `difflib` uses the
//!   Ratcliff/Obershelp algorithm; `similar`'s default is Myers. Both
//!   produce matching blocks in the same `(old_index, new_index, len)`
//!   shape, and for the realistic inputs langextract processes (mostly
//!   short medical / character / entity extractions) the results
//!   agree. Edge cases with many equal-length candidate matches may
//!   diverge by one block.
//!
//! # Minimal example
//!
//! ```no_run
//! use langextract_aligner::{align_extraction_groups, AlignmentOptions};
//! use langextract_core::{AlignmentStatus, Extraction};
//!
//! let source = "Patient is prescribed Naprosyn and prednisone for treatment.";
//! let groups = vec![
//!     vec![Extraction::new("medication", "Naprosyn")],
//!     vec![Extraction::new("medication", "prednisone")],
//! ];
//! let aligned = align_extraction_groups(groups, source, &AlignmentOptions::default()).unwrap();
//! for group in &aligned {
//!     for ex in group {
//!         assert_eq!(ex.alignment_status, Some(AlignmentStatus::MatchExact));
//!     }
//! }
//! ```

#![forbid(unsafe_code)]

pub mod align;
pub mod error;
pub mod fuzzy;
pub mod normalize;

pub use crate::align::{
    DEFAULT_DELIMITER, DEFAULT_FUZZY_THRESHOLD, AlignmentOptions, align_extraction_groups,
    align_extraction_groups_with,
};
pub use crate::error::AlignError;
pub use crate::normalize::{
    default_lowercase_tokens, lowercase_tokens, lowercase_tokens_from, normalize_token,
    normalize_tokens,
};
