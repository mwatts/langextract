//! Error types for `langextract-format`.

/// Errors returned by [`FormatHandler::parse_output`](crate::FormatHandler::parse_output)
/// and related operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum FormatError {
    /// The input was empty or had no parseable content.
    #[error("empty or invalid input string")]
    EmptyInput,

    /// A fenced block was required but none was found.
    #[error("input contains no valid {format} code fence")]
    FenceNotFound {
        /// The format type that was being searched for (`"json"` / `"yaml"`).
        format: &'static str,
    },

    /// Input contained multiple fenced blocks but the handler is
    /// configured to accept exactly one.
    #[error("multiple fenced blocks found; expected exactly one")]
    MultipleFencedBlocks,

    /// JSON / YAML parsing failed.
    #[error("failed to parse {format} content: {source}")]
    Parse {
        /// Which format failed (`"json"` or `"yaml"`).
        format: &'static str,
        /// The underlying parser error message.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync + 'static>,
    },

    /// The parsed content had the wrong shape — a wrapper key was
    /// expected but missing, an item was not a mapping, etc.
    #[error("{0}")]
    InvalidShape(String),
}

impl FormatError {
    /// Construct an [`FormatError::InvalidShape`] with a static message.
    #[must_use]
    pub fn shape(msg: impl Into<String>) -> Self {
        Self::InvalidShape(msg.into())
    }
}
