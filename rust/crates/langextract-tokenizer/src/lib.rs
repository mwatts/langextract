//! Tokenization utilities for the Rust port of
//! [langextract](https://github.com/google/langextract).
//!
//! Port of `langextract/core/tokenizer.py`. This crate provides:
//!
//! - [`Token`], [`TokenType`], [`TokenInterval`], [`TokenizedText`] — the
//!   token data model shared with the rest of the pipeline.
//! - The [`Tokenizer`] trait and its default implementation,
//!   [`RegexTokenizer`], which splits text into words, numbers, and
//!   punctuation without any regex-engine dependency.
//! - [`tokens_text`] for reconstructing the source substring spanned by a
//!   token interval.
//! - [`find_sentence_range`] for locating sentence boundaries — the
//!   primitive the chunking layer builds on.
//!
//! # Byte offsets, not character indices
//!
//! Every position stored in a [`Token::char_interval`] is a **byte offset
//! into the source `&str`**, not a Unicode code-point index. This matches
//! Rust's `&str` indexing convention and lets callers do
//! `&text[tok.char_interval.start..tok.char_interval.end]` as a safe
//! O(1) slice. The Python port of this module uses code-point indices;
//! consumers crossing the Python/Rust boundary need to convert.
//!
//! # What is *not* ported (yet)
//!
//! The Python module also ships a `UnicodeTokenizer` built on the third-
//! party `regex` package's `\X` grapheme-cluster support and Unicode
//! script properties. That implementation is deferred to a follow-up
//! crate because it requires `unicode-segmentation` and script detection
//! that are non-trivial to port. The [`RegexTokenizer`] here covers the
//! same ground as the Python default and is what the existing pipeline
//! uses for English text.
//!
//! # Preparing for fuzzy alignment
//!
//! The upcoming resolver port will build its character-interval fuzzy
//! alignment on top of these tokens. Specifically, the resolver needs:
//!
//! 1. Deterministic, byte-accurate [`Token::char_interval`]s so that a
//!    matched token span can be mapped back to the source text without
//!    re-scanning.
//! 2. A sentence-boundary oracle ([`find_sentence_range`]) for restricting
//!    alignment search to a reasonable window around each extraction.
//! 3. Stable tokenization across repeated calls — [`RegexTokenizer`] is
//!    pure and zero-state, so the same input always yields the same
//!    output. Do **not** cache tokenizations with an instance that holds
//!    mutable state; there is no such instance.

#![forbid(unsafe_code)]

pub mod error;
pub mod regex_tokenizer;
pub mod sentence;
pub mod types;

pub use crate::error::TokenizerError;
pub use crate::regex_tokenizer::RegexTokenizer;
pub use crate::sentence::{
    find_sentence_range, find_sentence_range_with, tokens_text, DEFAULT_ABBREVIATIONS,
};
pub use crate::types::{Token, TokenInterval, TokenType, TokenizedText};

/// A tokenizer: any type that can split text into [`Token`]s.
///
/// This is intentionally a simple synchronous trait; tokenization is pure
/// CPU work with no IO, so there is nothing to `.await`. Implementations
/// must be side-effect-free so that the same input always produces the
/// same output.
pub trait Tokenizer {
    /// Split `text` into tokens.
    fn tokenize(&self, text: &str) -> TokenizedText;
}

/// Convenience free function: tokenize with the default [`RegexTokenizer`].
///
/// Mirrors the Python `tokenize()` top-level helper.
#[must_use]
pub fn tokenize(text: &str) -> TokenizedText {
    RegexTokenizer::new().tokenize(text)
}
