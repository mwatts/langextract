//! Tokenizer error types.
//!
//! Port of the tokenizer-specific error classes in `core/tokenizer.py`.

/// Base error type for this crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum TokenizerError {
    /// A [`TokenInterval`](crate::TokenInterval) was out of range or
    /// malformed (start past end, or end past the token count).
    #[error("invalid token interval: start={start}, end={end}, total_tokens={total}")]
    InvalidTokenInterval {
        /// Start index that was supplied.
        start: usize,
        /// End index that was supplied.
        end: usize,
        /// Total number of tokens in the [`TokenizedText`](crate::TokenizedText).
        total: usize,
    },

    /// The starting token index for a sentence search was out of range.
    #[error("sentence start index {start} out of range (total tokens: {total})")]
    SentenceStartOutOfRange {
        /// Start index that was supplied.
        start: usize,
        /// Total number of tokens.
        total: usize,
    },
}
