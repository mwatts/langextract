//! # langextract (Rust)
//!
//! Top-level facade for the Rust port of
//! [langextract](https://github.com/google/langextract) — a library
//! for extracting structured data from language models with precise
//! source grounding. This crate composes the focused
//! `langextract-*` crates into a single [`extract`] function plus a
//! curated set of public re-exports.
//!
//! # Quick start
//!
//! ```ignore
//! use std::sync::Arc;
//! use langextract::{extract, ExtractRequest};
//! use langextract_core::{ExampleData, Extraction};
//!
//! # async fn demo() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
//! let model = /* your Arc<dyn LanguageModel> — see langextract_core::cli_adapter */;
//! # let model: Arc<dyn langextract_core::LanguageModel> = unimplemented!();
//!
//! let result = extract(
//!     model.as_ref(),
//!     ExtractRequest {
//!         text: "Alice is an engineer. Bob is a manager.".to_owned(),
//!         description: "Extract people and their roles.".to_owned(),
//!         examples: vec![ExampleData::new(
//!             "Carol is a doctor.",
//!             vec![Extraction::new("person", "Carol")],
//!         )],
//!         ..Default::default()
//!     },
//! ).await?;
//!
//! for ex in &result.extractions {
//!     println!("{} ({:?}): {}", ex.extraction_class, ex.char_interval, ex.extraction_text);
//! }
//! # Ok(()) }
//! ```
//!
//! # CLI-backed models
//!
//! If your "LLM" is a coding-agent CLI (Claude Code, aider,
//! gemini-cli, codex, a homegrown tool, …), implement
//! [`langextract_core::cli_adapter::CliRunner`] — a single async
//! method that shells out to your CLI and returns stdout — and wrap
//! it in
//! [`langextract_core::cli_adapter::CliLanguageModel`]. The
//! resulting value implements [`langextract_core::LanguageModel`]
//! and can be passed straight into [`extract`].
//!
//! See the `cli_adapter` module docs in `langextract-core` for the
//! gotchas: fenced-block extraction, concurrency cap, stdin-vs-argv,
//! non-zero-exit handling, timeouts, and so on.
//!
//! # What this crate is not
//!
//! - **Not a parser for specific document formats.** It takes a
//!   `String` of text. Convert your PDFs, HTML, or docx upstream
//!   before calling [`extract`].
//! - **Not a model provider.** It invokes whatever
//!   [`LanguageModel`](langextract_core::LanguageModel) you pass in
//!   but does not ship Gemini/`OpenAI`/Ollama glue. Use the
//!   `cli-adapter` for shell-based agents, or build your own
//!   provider.

#![forbid(unsafe_code)]

pub mod batch;
pub mod cache;
pub mod checkpoint;
pub mod error;
pub mod pipeline;
pub mod report;
pub mod request;
pub mod retry;

pub use crate::batch::{BatchItem, BatchOptions, extract_batch};
pub use crate::cache::{CacheKey, ChunkCache, InMemoryChunkCache, NoOpChunkCache};
pub use crate::checkpoint::{
    Checkpoint, CheckpointId, InMemoryCheckpoint, JsonlCheckpoint, NoOpCheckpoint,
};
pub use crate::error::ExtractError;
pub use crate::pipeline::{extract, extract_with_report};
pub use crate::report::{
    AlignmentCounts, ChunkReport, DocumentHealthThresholds, DocumentReport, HealthStatus,
    unaligned_reason_label,
};
pub use crate::request::{DEFAULT_INDEX_SUFFIX, DEFAULT_MAX_CHAR_BUFFER, ExtractRequest};
pub use crate::retry::{RetryPolicy, is_transient_format, is_transient_infer};

// ---- curated re-exports from the focused crates ----
//
// These let consumers `use langextract::{Extraction, ExampleData,
// FormatHandler, CliRunner, ...}` from one place without having to
// know the workspace structure.

pub use langextract_core::{
    ATTRIBUTE_SUFFIX, AlignmentStatus, AnnotatedDocument, AttributeMap, AttributeValue,
    BoxError, CharInterval, Document, DocumentId, EXTRACTIONS_KEY, ExampleData, Extraction,
    FormatType, InferError, InferenceParams, LanguageModel, Result as LangExtractResult,
    ScoredOutput, SchemaSupport, TokenInterval,
};

#[cfg(feature = "cli-adapter")]
pub use langextract_core::cli_adapter::{
    CliLanguageModel, CliLanguageModelBuilder, CliRunner, FencePolicy,
    extract_first_fenced_block, extract_last_fenced_block,
};
#[cfg(feature = "cli-adapter")]
pub use langextract_core::CliRunnerError;

pub use langextract_format::{FormatError, FormatHandler, FormatHandlerBuilder};
pub use langextract_prompting::{
    PromptBuilder, PromptTemplateStructured, QAPromptGenerator, StatelessPromptBuilder,
    ContextAwarePromptBuilder,
};
pub use langextract_chunking::{ChunkIterator, TextChunk};
pub use langextract_aligner::{
    AlignmentOptions, AlignmentReport, DEFAULT_FUZZY_THRESHOLD, FuzzySafeguards, UnalignedReason,
};
pub use langextract_tokenizer::{RegexTokenizer, Token, TokenType, TokenizedText, Tokenizer};
