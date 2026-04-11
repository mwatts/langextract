//! Error type for the top-level pipeline.
//!
//! Flattens the error types from every sub-crate into a single enum
//! so callers can `?`-chain against one error type instead of
//! juggling six.

use langextract_aligner::AlignError;
use langextract_chunking::ChunkingError;
use langextract_core::InferError;
use langextract_format::FormatError;
use langextract_prompting::PromptError;

/// Union of every error an `extract` call can produce.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum ExtractError {
    /// Failure while building a prompt or loading a template.
    #[error(transparent)]
    Prompt(#[from] PromptError),

    /// Failure while chunking the source document.
    #[error(transparent)]
    Chunking(#[from] ChunkingError),

    /// Failure while calling the language model.
    #[error(transparent)]
    Inference(#[from] InferError),

    /// Failure while parsing the LLM's output or walking its records.
    #[error(transparent)]
    Format(#[from] FormatError),

    /// Failure while aligning extractions to the source.
    #[error(transparent)]
    Alignment(#[from] AlignError),
}
