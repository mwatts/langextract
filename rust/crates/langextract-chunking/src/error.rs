//! Error types for the chunking crate.

/// Errors returned by chunking operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ChunkingError {
    /// A token interval was constructed with `start_index >= end_index`,
    /// or `start_index < 0`, or other out-of-range values.
    #[error("invalid token interval: start={start_index}, end={end_index}")]
    InvalidTokenInterval {
        /// Supplied start index.
        start_index: usize,
        /// Supplied end index.
        end_index: usize,
    },

    /// A chunk iterator expected `max_char_buffer >= 1`.
    #[error("max_char_buffer must be >= 1, got {0}")]
    InvalidBufferSize(usize),

    /// The tokenizer crate failed while helping reconstruct token text.
    #[error(transparent)]
    Tokenizer(#[from] langextract_tokenizer::TokenizerError),
}
