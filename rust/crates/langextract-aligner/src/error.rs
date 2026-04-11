//! Error types for the aligner crate.

/// Errors returned by [`align_extraction_groups`](crate::align_extraction_groups).
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum AlignError {
    /// The internal delimiter used to join extraction texts for batched
    /// matching appeared in one of the extraction texts. If the delimiter
    /// did not collide, alignment would silently corrupt; this is a hard
    /// error.
    #[error("delimiter {delimiter:?} appears inside extraction text {extraction_text:?}")]
    DelimiterInExtraction {
        /// The delimiter that collided.
        delimiter: String,
        /// The extraction whose text contained the delimiter.
        extraction_text: String,
    },

    /// The configured delimiter does not tokenize to exactly one token
    /// under the active tokenizer. Callers should not normally hit this;
    /// the default delimiter (Unicode unit separator, U+241F) is chosen
    /// to tokenize as a single punctuation token under
    /// [`RegexTokenizer`](langextract_tokenizer::RegexTokenizer).
    #[error("delimiter {delimiter:?} must tokenize to exactly one token, got {count}")]
    InvalidDelimiter {
        /// The misconfigured delimiter.
        delimiter: String,
        /// The number of tokens it produced.
        count: usize,
    },
}

