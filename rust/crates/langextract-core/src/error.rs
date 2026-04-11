//! Error types for `langextract-core`.
//!
//! This crate follows the Microsoft Rust guideline `M-ERRORS-CANONICAL-STRUCTS`:
//! errors are concrete structs/enums implementing [`std::error::Error`], never
//! `Box<dyn Error>` in public API.
//!
//! The top-level [`LangExtractError`] is the union of every fallible operation
//! the core crate exposes. Individual modules have their own narrower error
//! types ([`InferError`], [`CliRunnerError`]) which convert into it via `?`.

use std::fmt;

/// Base error type for the `langextract-core` crate.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum LangExtractError {
    /// An inference call through a [`LanguageModel`](crate::model::LanguageModel)
    /// failed.
    #[error(transparent)]
    Inference(#[from] InferError),

    /// A CLI-backed language model runner failed.
    #[error(transparent)]
    CliRunner(#[from] CliRunnerError),

    /// The LLM output could not be parsed into structured extractions.
    #[error("failed to parse LLM output: {0}")]
    Parse(String),

    /// An alignment / resolution step failed.
    #[error("failed to resolve extraction against source text: {0}")]
    Resolve(String),
}

/// Error returned by [`LanguageModel::infer`](crate::model::LanguageModel::infer).
///
/// Providers should map their native errors into one of the variants below
/// rather than leaking SDK-specific types.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum InferError {
    /// The prompt batch was rejected by the provider (empty, too large, etc.).
    #[error("invalid prompt batch: {0}")]
    InvalidBatch(String),

    /// The provider responded but the response was malformed.
    #[error("malformed provider response: {0}")]
    MalformedResponse(String),

    /// The provider returned no completions for at least one prompt.
    ///
    /// The pipeline treats this as a recoverable error: the chunk will be
    /// surfaced to the resolver with empty output and typically downgraded
    /// to a warning rather than aborting the whole document.
    #[error("provider returned no completions")]
    EmptyCompletions,

    /// Rate-limited, quota exceeded, or temporarily unavailable.
    #[error("provider temporarily unavailable: {0}")]
    Transient(String),

    /// Any other provider-side failure. Wraps the provider's native error.
    #[error("provider error: {0}")]
    Provider(#[source] BoxError),
}

/// Error returned by a [`CliRunner`](crate::cli_adapter::CliRunner) impl.
///
/// Intentionally narrow: the adapter needs to distinguish "the binary never
/// ran" from "the binary ran but returned non-zero" so that the pipeline can
/// retry the latter and abort the former.
#[cfg(feature = "cli-adapter")]
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum CliRunnerError {
    /// The CLI binary could not be located or could not be spawned.
    #[error("failed to spawn CLI: {0}")]
    Spawn(String),

    /// The CLI ran but exited with a non-zero status code.
    ///
    /// `stderr` is captured so the pipeline can log it. The pipeline treats
    /// this as retriable by default.
    #[error("CLI exited with status {status}: {stderr}")]
    NonZeroExit {
        /// The process's exit status code, if available.
        status: i32,
        /// Contents of the process's stderr stream (lossy UTF-8).
        stderr: String,
    },

    /// The CLI ran successfully but produced output that the adapter could
    /// not interpret (e.g., no fenced block found when one was required).
    #[error("CLI output could not be interpreted: {0}")]
    UnparseableOutput(String),

    /// IO error while communicating with the subprocess (broken pipe,
    /// truncated stdin, etc.).
    #[error("IO error talking to CLI: {0}")]
    Io(#[source] BoxError),

    /// The CLI did not complete within the configured timeout.
    #[error("CLI timed out after {seconds}s")]
    Timeout {
        /// Configured timeout in seconds.
        seconds: u64,
    },
}

/// Stub kept so that builds without the `cli-adapter` feature still have a
/// named type for error-enum discriminants from users who depend on us
/// transitively. It is uninhabited and never constructed.
#[cfg(not(feature = "cli-adapter"))]
#[derive(Debug, thiserror::Error)]
#[error("cli-adapter feature disabled")]
pub enum CliRunnerError {}

/// A boxed, thread-safe, `Send + Sync` error used to wrap foreign errors
/// without leaking their concrete type through the public API.
pub type BoxError = Box<dyn std::error::Error + Send + Sync + 'static>;

impl InferError {
    /// Convenience constructor for wrapping a foreign error as [`InferError::Provider`].
    pub fn provider<E>(err: E) -> Self
    where
        E: std::error::Error + Send + Sync + 'static,
    {
        Self::Provider(Box::new(err))
    }
}

/// Shorthand `Result` type using [`LangExtractError`].
pub type Result<T, E = LangExtractError> = std::result::Result<T, E>;

// Guarantee these error types satisfy `Send + Sync + 'static` so they can
// cross task boundaries in an async pipeline. Compile-time assertion only.
const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<LangExtractError>();
    assert_send_sync::<InferError>();
};

impl fmt::Display for crate::model::SchemaSupport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::None => f.write_str("none"),
            Self::FencedJson => f.write_str("fenced-json"),
            Self::FencedYaml => f.write_str("fenced-yaml"),
        }
    }
}
