//! Error type for the prompting crate.

/// Errors returned by prompt construction, template loading, or
/// few-shot example formatting.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum PromptError {
    /// A downstream [`langextract_format`] operation failed —
    /// typically while formatting a few-shot example into its
    /// JSON/YAML text form.
    #[error(transparent)]
    Format(#[from] langextract_format::FormatError),

    /// Failed to read a prompt-template file from disk.
    #[error("failed to read prompt template from {path}: {source}")]
    Read {
        /// Path that was being read.
        path: String,
        /// Underlying IO error.
        #[source]
        source: std::io::Error,
    },

    /// Failed to deserialize a prompt template. Covers both the YAML
    /// and JSON paths.
    #[error("failed to parse prompt template: {0}")]
    Parse(String),
}
