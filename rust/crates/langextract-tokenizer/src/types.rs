//! Token data model.
//!
//! Port of the types in `langextract/core/tokenizer.py`.
//!
//! The character interval on each [`Token`] reuses
//! [`langextract_core::CharInterval`] — the tokenizer does not define its own
//! `CharInterval` type, unlike the Python module which declares one locally
//! for historical reasons. All character positions are **byte offsets into
//! the original `&str`**, not Unicode code-point indices. This is the native
//! Rust convention and allows `&text[interval.start..interval.end]` to be a
//! safe, O(1) slice.

use langextract_core::CharInterval;
use serde::{Deserialize, Serialize};

/// Classification of a token produced by a [`Tokenizer`](crate::Tokenizer).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum TokenType {
    /// A run of alphabetic characters (`[^\W\d_]+` in the Python source).
    Word,
    /// A run of decimal digit characters.
    Number,
    /// A symbol or punctuation character, or a run of the same symbol
    /// (e.g. `!!!`). Underscore is classified as punctuation to match the
    /// Python tokenizer.
    Punctuation,
}

/// A single token located in some source text.
///
/// Port of `Token` in `core/tokenizer.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Token {
    /// Zero-based position of this token in the token sequence.
    pub index: usize,

    /// What kind of token this is.
    pub token_type: TokenType,

    /// Byte offsets into the source text that this token spans.
    pub char_interval: CharInterval,

    /// `true` if the whitespace gap before this token contained a `\n` or
    /// `\r` character. Used by [`find_sentence_range`](crate::find_sentence_range)
    /// to detect paragraph-style sentence breaks.
    pub first_token_after_newline: bool,
}

/// A half-open interval `[start_index, end_index)` over a token sequence.
///
/// Port of `TokenInterval`. Used by the resolver to describe which tokens
/// in a [`TokenizedText`] are covered by an extraction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TokenInterval {
    /// Index of the first token in the interval.
    pub start_index: usize,
    /// Index one past the last token in the interval.
    pub end_index: usize,
}

impl TokenInterval {
    /// Construct a new interval.
    #[must_use]
    pub const fn new(start_index: usize, end_index: usize) -> Self {
        Self {
            start_index,
            end_index,
        }
    }

    /// Number of tokens in the interval.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end_index - self.start_index
    }

    /// Whether the interval is empty.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start_index == self.end_index
    }
}

/// The output of a [`Tokenizer`](crate::Tokenizer) run: the original text
/// plus the tokens extracted from it.
///
/// Port of `TokenizedText`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct TokenizedText {
    /// The exact source text that was tokenized. Byte offsets in each
    /// [`Token::char_interval`] index into this string.
    pub text: String,

    /// The tokens, in order.
    pub tokens: Vec<Token>,
}

impl TokenizedText {
    /// Construct a new empty tokenization of the given text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            tokens: Vec::new(),
        }
    }

    /// Number of tokens.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    /// Whether no tokens were produced.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Borrow the source text slice covered by a single token.
    ///
    /// Returns `None` if the token's interval is out of bounds for the text
    /// (should not happen for tokens this crate produces).
    #[must_use]
    pub fn token_text(&self, token: &Token) -> Option<&str> {
        self.text
            .get(token.char_interval.start..token.char_interval.end)
    }
}
