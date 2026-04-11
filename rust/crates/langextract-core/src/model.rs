//! The [`LanguageModel`] trait and its supporting types.
//!
//! This module is the central abstraction of the crate. Every LLM backend —
//! whether a hosted API (Gemini, `OpenAI`), a local server (Ollama), or a
//! command-line coding agent (see [`crate::cli_adapter`]) — implements
//! [`LanguageModel`]. The extraction pipeline then holds an
//! `Arc<dyn LanguageModel>` and invokes it on batches of chunked prompts.
//!
//! # Design notes
//!
//! - **Dyn-compatible.** The trait uses `#[async_trait::async_trait]` so that
//!   `Arc<dyn LanguageModel>` works. The small allocation cost per call is
//!   irrelevant next to the cost of an actual LLM invocation.
//! - **Batched.** [`LanguageModel::infer`] takes a slice of prompts and
//!   returns one [`Vec<ScoredOutput>`] per input. Providers that support
//!   native batching (Gemini, `OpenAI`) can submit them together; providers
//!   that don't (Ollama, most CLIs) loop internally with bounded concurrency.
//! - **Schema support as data, not polymorphism.** The pipeline inspects
//!   [`LanguageModel::schema_support`] to decide whether it needs to embed
//!   format hints into the prompt itself. A CLI backend typically returns
//!   [`SchemaSupport::None`].

use async_trait::async_trait;

use crate::error::InferError;

/// What kind of structured output support a provider offers.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[non_exhaustive]
pub enum SchemaSupport {
    /// The provider has no native controlled-generation feature. The
    /// pipeline will embed the expected output format into the prompt text
    /// and rely on the resolver to parse fenced code blocks out of the
    /// free-form response. **This is the right choice for CLI providers.**
    #[default]
    None,

    /// The provider accepts a JSON schema and guarantees the response will
    /// conform to it. Output will be a single fenced code block tagged
    /// `json`.
    FencedJson,

    /// The provider accepts a YAML schema. Output will be a single fenced
    /// code block tagged `yaml`.
    FencedYaml,
}

/// Generation parameters passed on each [`LanguageModel::infer`] call.
///
/// This is intentionally minimal. Provider-specific knobs should be
/// configured when the `LanguageModel` impl is constructed, not passed
/// per-call — the pipeline reuses the same model across every chunk and
/// only varies the prompt.
#[derive(Debug, Clone, Default)]
pub struct InferenceParams {
    /// Sampling temperature (0.0 = deterministic).
    pub temperature: Option<f32>,

    /// Maximum tokens the provider may emit per completion.
    pub max_output_tokens: Option<u32>,

    /// Stop sequences. If the provider encounters any of these, it halts
    /// generation. Not all providers honor this.
    pub stop_sequences: Vec<String>,
}

/// A single completion from an LLM, optionally with a score.
///
/// The score is provider-defined: log-probability, a confidence, or `None`
/// if the provider doesn't expose one. Most CLI providers return `None`.
#[derive(Debug, Clone)]
pub struct ScoredOutput {
    /// The completion text as returned by the provider.
    ///
    /// For providers with [`SchemaSupport::None`], this text may include
    /// fence markers and surrounding prose; the resolver is responsible for
    /// extracting the actual structured payload.
    pub output: String,

    /// Provider-defined score, if available.
    pub score: Option<f32>,
}

impl ScoredOutput {
    /// Construct an unscored output.
    #[must_use]
    pub fn unscored(output: impl Into<String>) -> Self {
        Self {
            output: output.into(),
            score: None,
        }
    }
}

/// The core provider trait.
///
/// # Implementing
///
/// Any type that implements this trait can be plugged into the extraction
/// pipeline. The implementation must be `Send + Sync` because the pipeline
/// shares it across tasks via `Arc`.
///
/// ```no_run
/// use async_trait::async_trait;
/// use langextract_core::{InferError, InferenceParams, LanguageModel, ScoredOutput, SchemaSupport};
///
/// struct EchoModel;
///
/// #[async_trait]
/// impl LanguageModel for EchoModel {
///     async fn infer(
///         &self,
///         prompts: &[String],
///         _params: &InferenceParams,
///     ) -> Result<Vec<Vec<ScoredOutput>>, InferError> {
///         Ok(prompts
///             .iter()
///             .map(|p| vec![ScoredOutput::unscored(p.clone())])
///             .collect())
///     }
///
///     fn schema_support(&self) -> SchemaSupport {
///         SchemaSupport::None
///     }
/// }
/// ```
#[async_trait]
pub trait LanguageModel: Send + Sync {
    /// Run inference on a batch of prompts.
    ///
    /// The returned vector must have the same length as `prompts`. Each
    /// inner vector is a list of candidates for that prompt (usually just
    /// one). An empty inner vector must be reported as
    /// [`InferError::EmptyCompletions`] — return a real error rather than
    /// silently losing data.
    async fn infer(
        &self,
        prompts: &[String],
        params: &InferenceParams,
    ) -> Result<Vec<Vec<ScoredOutput>>, InferError>;

    /// What structured-output support this provider offers.
    ///
    /// Defaults to [`SchemaSupport::None`] — safe for any provider that
    /// can't guarantee format compliance.
    fn schema_support(&self) -> SchemaSupport {
        SchemaSupport::None
    }
}

// Compile-time assertion that `dyn LanguageModel` is object-safe and
// `Send + Sync`. This catches accidents where a method signature breaks
// dyn-compat (e.g., adding a generic method).
const _: fn() = || {
    const fn assert_dyn_compat<T: ?Sized + Send + Sync>() {}
    assert_dyn_compat::<dyn LanguageModel>();
};

#[cfg(test)]
mod tests {
    use super::*;

    struct Dummy;

    #[async_trait]
    impl LanguageModel for Dummy {
        async fn infer(
            &self,
            prompts: &[String],
            _params: &InferenceParams,
        ) -> Result<Vec<Vec<ScoredOutput>>, InferError> {
            Ok(prompts
                .iter()
                .map(|p| vec![ScoredOutput::unscored(p.clone())])
                .collect())
        }
    }

    #[tokio::test]
    async fn dummy_model_round_trip() {
        use std::sync::Arc;
        let m: Arc<dyn LanguageModel> = Arc::new(Dummy);
        let out = m
            .infer(&["hello".into(), "world".into()], &InferenceParams::default())
            .await
            .unwrap();
        assert_eq!(out.len(), 2);
        assert_eq!(out[0][0].output, "hello");
        assert_eq!(out[1][0].output, "world");
    }
}
