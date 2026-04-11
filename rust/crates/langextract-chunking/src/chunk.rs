//! The [`TextChunk`] type and helpers for computing chunk-level
//! intervals from a tokenized document.

use langextract_core::{CharInterval, DocumentId, TokenInterval};
use langextract_tokenizer::{TokenizedText, tokens_text};

use crate::error::ChunkingError;

/// A chunk of source text ready to be fed to a language model.
///
/// Port of `TextChunk` from `langextract/chunking.py`. The Python
/// version uses lazy properties backed by a source-document reference;
/// the Rust port computes everything eagerly at chunk-emission time
/// and stores the chunk text by value. This trades a little memory for
/// simpler ownership — the chunk is self-contained and can be moved or
/// cloned across async tasks without worrying about the source
/// document's lifetime.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TextChunk {
    /// Token span of this chunk within the source document.
    pub token_interval: TokenInterval,

    /// Character span of this chunk within the source document.
    ///
    /// Byte offsets (not code-point indices), matching the convention
    /// established by the tokenizer crate.
    pub char_interval: CharInterval,

    /// The text slice this chunk covers, owned.
    pub text: String,

    /// Identifier of the source document, if one was supplied to the
    /// chunk iterator.
    pub document_id: Option<DocumentId>,

    /// Optional per-document context (from
    /// [`langextract_core::Document::additional_context`]) that the
    /// pipeline can inject into the prompt alongside this chunk.
    pub additional_context: Option<String>,
}

impl TextChunk {
    /// Convenience accessor: length of the chunk text in bytes.
    #[must_use]
    pub fn len(&self) -> usize {
        self.text.len()
    }

    /// Convenience: whether the chunk is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Collapse all whitespace in the chunk text to single spaces —
    /// suitable for injection into a single-line prompt slot without
    /// introducing gratuitous newlines.
    ///
    /// Port of the `_sanitize` helper in `chunking.py`.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkingError::InvalidTokenInterval`] with a synthetic
    /// interval if the chunk contains only whitespace — this matches
    /// Python's `ValueError` but keeps the Rust error taxonomy small.
    pub fn sanitized_text(&self) -> Result<String, ChunkingError> {
        let sanitized = sanitize_whitespace(&self.text);
        if sanitized.is_empty() {
            return Err(ChunkingError::InvalidTokenInterval {
                start_index: self.token_interval.start_index,
                end_index: self.token_interval.end_index,
            });
        }
        Ok(sanitized)
    }
}

/// Construct a [`TokenInterval`] with the chunking crate's validation
/// rules (start < end, both non-negative since they're `usize`).
///
/// Port of `create_token_interval`.
///
/// # Errors
///
/// Returns [`ChunkingError::InvalidTokenInterval`] if `start >= end`.
pub const fn create_token_interval(
    start_index: usize,
    end_index: usize,
) -> Result<TokenInterval, ChunkingError> {
    if start_index >= end_index {
        return Err(ChunkingError::InvalidTokenInterval {
            start_index,
            end_index,
        });
    }
    Ok(TokenInterval::new(start_index, end_index))
}

/// Get the source-text substring that corresponds to a token interval.
///
/// Port of `get_token_interval_text`. Thin wrapper around
/// [`langextract_tokenizer::tokens_text`] with the crate's error type.
///
/// # Errors
///
/// Returns [`ChunkingError::Tokenizer`] if the interval is invalid.
pub fn get_token_interval_text(
    tokenized: &TokenizedText,
    interval: TokenInterval,
) -> Result<&str, ChunkingError> {
    Ok(tokens_text(tokenized, interval)?)
}

/// Compute the character interval covering a token interval.
///
/// Port of `get_char_interval`.
///
/// # Errors
///
/// Returns [`ChunkingError::InvalidTokenInterval`] if the interval is
/// malformed or out of range.
pub fn get_char_interval(
    tokenized: &TokenizedText,
    interval: TokenInterval,
) -> Result<CharInterval, ChunkingError> {
    if interval.start_index >= interval.end_index {
        return Err(ChunkingError::InvalidTokenInterval {
            start_index: interval.start_index,
            end_index: interval.end_index,
        });
    }
    if interval.end_index > tokenized.tokens.len() {
        return Err(ChunkingError::InvalidTokenInterval {
            start_index: interval.start_index,
            end_index: interval.end_index,
        });
    }
    let start_token = &tokenized.tokens[interval.start_index];
    let end_token = &tokenized.tokens[interval.end_index - 1];
    Ok(CharInterval::new(
        start_token.char_interval.start,
        end_token.char_interval.end,
    ))
}

/// Collapse any run of whitespace to a single space and trim the ends.
fn sanitize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_ws = false;
    let mut seen_non_ws = false;
    for c in text.chars() {
        if c.is_whitespace() {
            if seen_non_ws {
                in_ws = true;
            }
        } else {
            if in_ws {
                out.push(' ');
            }
            out.push(c);
            in_ws = false;
            seen_non_ws = true;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn sanitize_collapses_whitespace() {
        assert_eq!(sanitize_whitespace("  hello\n\tworld  "), "hello world");
        assert_eq!(sanitize_whitespace("a   b\n\nc"), "a b c");
        assert_eq!(sanitize_whitespace("no_change"), "no_change");
        assert_eq!(sanitize_whitespace("   "), "");
    }

    #[test]
    fn create_token_interval_validates() {
        assert!(create_token_interval(0, 1).is_ok());
        assert!(create_token_interval(0, 0).is_err());
        assert!(create_token_interval(5, 3).is_err());
    }
}
