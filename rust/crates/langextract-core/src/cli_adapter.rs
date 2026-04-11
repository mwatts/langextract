//! CLI-backed [`LanguageModel`] adapter.
//!
//! This module exists specifically for the use case where your "LLM" is
//! actually a command-line coding agent (Claude Code, aider, gemini-cli,
//! codex, a homegrown tool, …) and you want to drive langextract through it
//! without writing a full provider from scratch.
//!
//! # The two-trait shape
//!
//! The adapter is deliberately split into two traits so that **you can
//! plug in an existing trait you already use for CLI management** without
//! rewriting it:
//!
//! ```text
//!     your trait (manages CLI processes, already tested)
//!         │
//!         │  one-line bridge: `impl CliRunner for YourAdapter`
//!         ▼
//!     CliRunner     ← narrow: "given a prompt, return stdout"
//!         │
//!         │  generic wrapper
//!         ▼
//!     CliLanguageModel<R>   ← implements langextract's LanguageModel trait
//!         │
//!         ▼
//!     extraction pipeline
//! ```
//!
//! [`CliRunner`] is the minimum possible contract: **one async method that
//! takes a prompt string and returns the CLI's stdout**. It makes no
//! assumptions about how you spawn the process, what binary you use,
//! whether you pool instances, or how you pass configuration — all of that
//! stays in your existing code. You only have to write a one-line `impl
//! CliRunner` that forwards to your existing trait.
//!
//! [`CliLanguageModel`] is the generic wrapper that implements
//! [`LanguageModel`] in terms of any `CliRunner`. It handles everything
//! langextract-specific: concurrency capping, fenced-block extraction from
//! the raw stdout, and mapping [`CliRunnerError`] into [`InferError`].
//!
//! # Gotchas — read these before wiring it up
//!
//! These are the things that routinely bite people when they point
//! langextract at a coding-agent CLI. The defaults below are tuned to
//! avoid each one, but you should understand them so you can adjust if
//! your CLI is unusual.
//!
//! 1. **The CLI must be coaxed into returning a fenced structured block.**
//!    langextract's resolver parses Markdown fenced code blocks tagged
//!    `json` or `yaml`. If your CLI wraps output in prose ("Here's the
//!    JSON you asked for:"), [`FencePolicy::Last`] (the default) will
//!    cleanly strip it back down to just the fenced payload. If the CLI
//!    emits multiple code blocks during reasoning, the **last** one is
//!    almost always the intended answer; hence the default. Use
//!    [`FencePolicy::First`] only if your CLI puts the answer first and
//!    then chats afterward.
//!
//! 2. **Set [`SchemaSupport::None`].** This is the default
//!    [`LanguageModel::schema_support`] for [`CliLanguageModel`] and you
//!    should not change it unless your CLI genuinely supports controlled
//!    generation. Reporting `None` tells the pipeline to embed the format
//!    hint into the prompt text rather than try to use a provider-native
//!    JSON schema feature (which no CLI has).
//!
//! 3. **Feed prompts via stdin, not argv.** Prompts easily exceed OS argv
//!    length limits and cause escaping nightmares (quotes, newlines, shell
//!    metacharacters). Your [`CliRunner`] impl should push the prompt
//!    through stdin. The adapter passes you a `&str` and trusts you to do
//!    this correctly.
//!
//! 4. **Non-zero exit is an error, not an empty success.** If your CLI
//!    exits non-zero, return [`CliRunnerError::NonZeroExit`] with stderr
//!    captured. Returning an empty `Ok("")` will make the resolver parse
//!    nothing, lower recall, and look like a "the model found nothing"
//!    result — which is almost always a bug, not a real finding.
//!
//! 5. **Cap concurrency.** Coding-agent CLIs are heavy: each invocation
//!    may spawn language-server processes, load context, or hit its own
//!    rate limits. [`CliLanguageModel::new`] takes a `concurrency` knob
//!    that's enforced via an internal semaphore. Default sensibly: **1 or
//!    2** for most coding agents. Ignore langextract's chunk-level
//!    parallelism if your CLI is single-threaded.
//!
//! 6. **Use larger chunks.** Offset (5) above by setting `max_char_buffer`
//!    in the pipeline higher than you would for a cheap API — fewer,
//!    bigger invocations are better when each invocation is slow. That's
//!    a pipeline configuration, not something this adapter sets.
//!
//! 7. **Surface the format hint explicitly.** Because
//!    [`SchemaSupport::None`] means the pipeline embeds the format hint
//!    into the prompt, you should let the pipeline's prompt builder do
//!    the work. Don't strip the format hint out of the prompt before
//!    passing it to the CLI — the resolver won't find the expected fence
//!    and will report a parse error.
//!
//! 8. **Reasoning/thinking interleaving.** Many coding agents stream
//!    "thinking" output interleaved with the final answer. If the CLI has
//!    a flag like `--print` / `--output-format=text` / `--quiet` that
//!    suppresses reasoning, use it. Otherwise [`FencePolicy::Last`]
//!    should still cope.
//!
//! 9. **Timeouts.** A stuck CLI will freeze the pipeline. Your [`CliRunner`]
//!    impl is responsible for timing out its own process spawn — the
//!    adapter doesn't inject a timeout because the right timeout depends
//!    entirely on your CLI.
//!
//! 10. **Idempotency / retries.** The pipeline may retry a failed chunk.
//!     Don't keep state between calls in your `CliRunner` unless you're
//!     deliberately pooling warm CLI instances. Each call should be
//!     independent.
//!
//! # Minimal example
//!
//! ```no_run
//! use async_trait::async_trait;
//! use langextract_core::cli_adapter::{CliLanguageModel, CliRunner};
//! use langextract_core::error::CliRunnerError;
//!
//! // 1. Your existing trait — already tested, already wired up.
//! #[async_trait]
//! trait MyAgentCli: Send + Sync {
//!     async fn prompt(&self, input: &str) -> Result<String, std::io::Error>;
//! }
//!
//! // 2. One-line bridge from your trait into `CliRunner`.
//! struct Bridge<T>(T);
//!
//! #[async_trait]
//! impl<T: MyAgentCli> CliRunner for Bridge<T> {
//!     async fn run(&self, prompt: &str) -> Result<String, CliRunnerError> {
//!         self.0
//!             .prompt(prompt)
//!             .await
//!             .map_err(|e| CliRunnerError::Io(Box::new(e)))
//!     }
//! }
//!
//! // 3. Wrap the bridge in a CliLanguageModel and hand it to the pipeline.
//! fn build_model<T: MyAgentCli + 'static>(cli: T) -> CliLanguageModel<Bridge<T>> {
//!     CliLanguageModel::builder(Bridge(cli))
//!         .concurrency(1) // heavy CLIs → low concurrency
//!         .build()
//! }
//! ```

use std::sync::Arc;

use async_trait::async_trait;
use tokio::sync::Semaphore;

use crate::error::{CliRunnerError, InferError};
use crate::model::{InferenceParams, LanguageModel, SchemaSupport, ScoredOutput};

/// A minimal async "run a prompt through a CLI" interface.
///
/// This is the trait you implement (or bridge your existing trait into).
/// See the [module-level docs](crate::cli_adapter) for design notes and
/// gotchas.
///
/// # Contract
///
/// - `run` takes the full prompt (already formatted by the pipeline, with
///   format hints embedded) and returns the CLI's stdout as a `String`.
/// - Stdout should be captured **losslessly**; the adapter will strip
///   fences itself based on [`FencePolicy`].
/// - **Non-zero exit must return [`CliRunnerError::NonZeroExit`]**, not
///   `Ok("")`. See gotcha #4 in the module docs.
/// - **Timeouts are the implementer's responsibility.** See gotcha #9.
#[async_trait]
pub trait CliRunner: Send + Sync {
    /// Run the CLI once with the given prompt and return its stdout.
    async fn run(&self, prompt: &str) -> Result<String, CliRunnerError>;
}

/// Policy controlling how the adapter extracts a structured payload from
/// the CLI's raw stdout.
///
/// Default: [`FencePolicy::Last`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[non_exhaustive]
pub enum FencePolicy {
    /// Pass the CLI's stdout through unchanged. Use this only if your CLI
    /// is already configured to emit exactly the structured payload (no
    /// prose, no markdown fences).
    Raw,
    /// Extract the first Markdown fenced code block found in the output.
    First,
    /// Extract the last Markdown fenced code block found in the output.
    /// **Default.** This is usually what you want for coding agents that
    /// reason first and then answer.
    #[default]
    Last,
}

/// Generic [`LanguageModel`] impl that delegates to any [`CliRunner`].
///
/// Construct via [`CliLanguageModel::builder`]. The runner is consumed by
/// value and kept in an `Arc` so calls from multiple concurrent tasks
/// share the same instance.
#[derive(Debug, Clone)]
pub struct CliLanguageModel<R: CliRunner> {
    runner: Arc<R>,
    fence_policy: FencePolicy,
    semaphore: Arc<Semaphore>,
}

impl<R: CliRunner> CliLanguageModel<R> {
    /// Start configuring a new adapter.
    #[must_use]
    pub fn builder(runner: R) -> CliLanguageModelBuilder<R> {
        CliLanguageModelBuilder {
            runner,
            fence_policy: FencePolicy::default(),
            concurrency: 1,
        }
    }
}

/// Builder for [`CliLanguageModel`] (`M-INIT-BUILDER`).
///
/// Heavy CLI providers have several independent knobs and the builder
/// pattern keeps construction readable.
#[derive(Debug)]
pub struct CliLanguageModelBuilder<R: CliRunner> {
    runner: R,
    fence_policy: FencePolicy,
    concurrency: usize,
}

impl<R: CliRunner> CliLanguageModelBuilder<R> {
    /// Override the fence-extraction policy. Default: [`FencePolicy::Last`].
    #[must_use]
    pub const fn fence_policy(mut self, policy: FencePolicy) -> Self {
        self.fence_policy = policy;
        self
    }

    /// Cap the number of concurrent CLI invocations. Default: `1`.
    ///
    /// Most coding agents should stay at 1 or 2. The pipeline will happily
    /// submit many prompts in parallel; this semaphore is what prevents
    /// your machine from thrashing or your CLI from rate-limiting itself.
    ///
    /// # Panics
    ///
    /// Panics if `concurrency == 0`.
    #[must_use]
    pub fn concurrency(mut self, concurrency: usize) -> Self {
        assert!(concurrency > 0, "concurrency must be >= 1");
        self.concurrency = concurrency;
        self
    }

    /// Finalize the builder.
    #[must_use]
    pub fn build(self) -> CliLanguageModel<R> {
        CliLanguageModel {
            runner: Arc::new(self.runner),
            fence_policy: self.fence_policy,
            semaphore: Arc::new(Semaphore::new(self.concurrency)),
        }
    }
}

#[async_trait]
impl<R: CliRunner + 'static> LanguageModel for CliLanguageModel<R> {
    async fn infer(
        &self,
        prompts: &[String],
        _params: &InferenceParams,
    ) -> Result<Vec<Vec<ScoredOutput>>, InferError> {
        // Build one future per prompt and drive them concurrently with
        // `try_join_all`. Concurrency is capped by the internal semaphore,
        // not by the runtime — this keeps the core crate runtime-agnostic.
        // Order is preserved because `try_join_all` returns results in the
        // order of the input futures.
        let futs = prompts.iter().map(|prompt| {
            let runner = Arc::clone(&self.runner);
            let sem = Arc::clone(&self.semaphore);
            let fence_policy = self.fence_policy;
            let prompt = prompt.clone();
            async move {
                // `acquire_owned` can only fail if the semaphore has been
                // closed, which we never do.
                let _permit = sem.acquire_owned().await.map_err(|e| {
                    InferError::Provider(Box::new(std::io::Error::other(format!(
                        "semaphore closed: {e}"
                    ))))
                })?;
                let raw = runner.run(&prompt).await.map_err(map_cli_error)?;
                let payload = apply_fence_policy(&raw, fence_policy)
                    .ok_or_else(|| {
                        InferError::MalformedResponse(
                            "CLI output contained no fenced code block".to_owned(),
                        )
                    })?
                    .to_owned();
                Ok::<Vec<ScoredOutput>, InferError>(vec![ScoredOutput::unscored(payload)])
            }
        });
        futures_util::future::try_join_all(futs).await
    }

    fn schema_support(&self) -> SchemaSupport {
        // Hard-coded to None: see gotcha #2 in the module docs.
        SchemaSupport::None
    }
}

/// Map a [`CliRunnerError`] to an [`InferError`] variant.
///
/// Transient failures (non-zero exit, IO, timeout) become
/// [`InferError::Transient`] so the pipeline can retry; unparseable output
/// and spawn failures become [`InferError::Provider`] which is fatal.
fn map_cli_error(err: CliRunnerError) -> InferError {
    match err {
        CliRunnerError::Spawn(msg) => {
            InferError::Provider(Box::new(std::io::Error::other(msg)))
        }
        CliRunnerError::NonZeroExit { status, stderr } => InferError::Transient(format!(
            "CLI exited with status {status}: {stderr}"
        )),
        CliRunnerError::UnparseableOutput(msg) => InferError::MalformedResponse(msg),
        CliRunnerError::Io(e) => InferError::Transient(format!("CLI IO error: {e}")),
        CliRunnerError::Timeout { seconds } => {
            InferError::Transient(format!("CLI timed out after {seconds}s"))
        }
    }
}

/// Extract a payload from a raw CLI stdout string according to a policy.
///
/// Returns `Some(&str)` referencing a slice of the input (no allocation)
/// or `None` if the policy requires a fence and none was found.
#[must_use]
pub fn apply_fence_policy(raw: &str, policy: FencePolicy) -> Option<&str> {
    match policy {
        FencePolicy::Raw => Some(raw.trim()),
        FencePolicy::First => extract_first_fenced_block(raw),
        FencePolicy::Last => extract_last_fenced_block(raw),
    }
}

/// Extract the **last** Markdown fenced code block in `text`, stripping
/// any language tag.
///
/// A "fenced block" is any text between two occurrences of triple-backtick.
/// The language tag (if present) is a single word immediately after the
/// opening fence and before the first newline (e.g. `json`).
///
/// Returns `None` if fewer than two triple-backticks are found.
#[must_use]
pub fn extract_last_fenced_block(text: &str) -> Option<&str> {
    let end_fence = text.rfind("```")?;
    let before_end = text.get(..end_fence)?;
    let start_fence = before_end.rfind("```")?;
    let inner = text.get(start_fence + 3..end_fence)?;
    Some(strip_language_tag(inner).trim_matches('\n'))
}

/// Extract the **first** Markdown fenced code block in `text`. See
/// [`extract_last_fenced_block`] for details.
#[must_use]
pub fn extract_first_fenced_block(text: &str) -> Option<&str> {
    let start_fence = text.find("```")?;
    let after_start = text.get(start_fence + 3..)?;
    let end_rel = after_start.find("```")?;
    let inner = after_start.get(..end_rel)?;
    Some(strip_language_tag(inner).trim_matches('\n'))
}

/// If the first line of `inner` looks like a language tag (a short
/// alphanumeric token), strip it off along with the trailing newline.
fn strip_language_tag(inner: &str) -> &str {
    let Some(first_line_end) = inner.find('\n') else {
        return inner;
    };
    let (tag, rest) = inner.split_at(first_line_end);
    // A language tag is a short token with no whitespace. If the "tag"
    // contains anything else, treat the whole thing as content.
    if !tag.is_empty()
        && tag.len() < 20
        && tag
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
    {
        // Skip the '\n' we found.
        &rest[1..]
    } else {
        inner
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn extract_last_strips_language_tag() {
        let text = "Here it is:\n```json\n{\"a\":1}\n```\nthanks!";
        assert_eq!(extract_last_fenced_block(text), Some("{\"a\":1}"));
    }

    #[test]
    fn extract_last_without_language_tag() {
        let text = "```\n{}\n```";
        assert_eq!(extract_last_fenced_block(text), Some("{}"));
    }

    #[test]
    fn extract_last_picks_final_of_multiple_blocks() {
        let text = "first:\n```yaml\nfoo: 1\n```\nand final:\n```json\n{\"b\":2}\n```";
        assert_eq!(extract_last_fenced_block(text), Some("{\"b\":2}"));
    }

    #[test]
    fn extract_first_picks_initial_block() {
        let text = "```json\n{\"a\":1}\n```\nand then:\n```yaml\nfoo: 2\n```";
        assert_eq!(extract_first_fenced_block(text), Some("{\"a\":1}"));
    }

    #[test]
    fn extract_returns_none_when_no_fence() {
        assert_eq!(extract_last_fenced_block("plain text"), None);
        assert_eq!(extract_first_fenced_block("plain text"), None);
    }

    #[test]
    fn extract_returns_none_with_single_fence() {
        // Only one triple-backtick — not a complete block.
        assert_eq!(extract_last_fenced_block("before ``` after"), None);
    }

    #[test]
    fn raw_policy_trims() {
        assert_eq!(apply_fence_policy("  hi  \n", FencePolicy::Raw), Some("hi"));
    }

    #[test]
    fn multiline_language_tag_is_treated_as_content() {
        // If the "tag" line contains a space, it's not a tag.
        let text = "```\nnot a tag\n{\"a\":1}\n```";
        // First line strip only fires on a pure alphanumeric token; here
        // the first line is "not a tag" which contains spaces, so the
        // whole thing is content.
        let extracted = extract_last_fenced_block(text).unwrap();
        assert!(extracted.contains("not a tag"));
        assert!(extracted.contains("{\"a\":1}"));
    }

    // --- Adapter smoke test with a fake CliRunner ---

    struct FakeRunner {
        // Scripted response.
        response: String,
    }

    #[async_trait]
    impl CliRunner for FakeRunner {
        async fn run(&self, _prompt: &str) -> Result<String, CliRunnerError> {
            Ok(self.response.clone())
        }
    }

    #[tokio::test]
    async fn cli_language_model_strips_fences_by_default() {
        let runner = FakeRunner {
            response: "sure thing:\n```json\n{\"x\":1}\n```\n".into(),
        };
        let model = CliLanguageModel::builder(runner).concurrency(2).build();
        let out = model
            .infer(&["prompt".into()], &InferenceParams::default())
            .await
            .unwrap();
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 1);
        assert_eq!(out[0][0].output, "{\"x\":1}");
    }

    #[tokio::test]
    async fn cli_language_model_missing_fence_is_malformed() {
        let runner = FakeRunner {
            response: "no fence here".into(),
        };
        let model = CliLanguageModel::builder(runner).build();
        let err = model
            .infer(&["prompt".into()], &InferenceParams::default())
            .await
            .unwrap_err();
        assert!(matches!(err, InferError::MalformedResponse(_)));
    }

    #[tokio::test]
    async fn raw_policy_passes_output_through() {
        let runner = FakeRunner {
            response: "{\"y\":2}".into(),
        };
        let model = CliLanguageModel::builder(runner)
            .fence_policy(FencePolicy::Raw)
            .build();
        let out = model
            .infer(&["prompt".into()], &InferenceParams::default())
            .await
            .unwrap();
        assert_eq!(out[0][0].output, "{\"y\":2}");
    }

    #[test]
    fn schema_support_is_none() {
        let runner = FakeRunner {
            response: String::new(),
        };
        let model = CliLanguageModel::builder(runner).build();
        assert_eq!(model.schema_support(), SchemaSupport::None);
    }
}
