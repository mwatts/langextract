//! Core data types and traits for the Rust port of
//! [langextract](https://github.com/google/langextract).
//!
//! This crate contains **only** the pieces every other layer of the library
//! will depend on:
//!
//! - Domain data types ([`Extraction`], [`Document`], [`AnnotatedDocument`],
//!   [`ExampleData`], …) ported from `langextract/core/data.py`.
//! - Canonical error types ([`LangExtractError`] and friends) following
//!   `M-ERRORS-CANONICAL-STRUCTS`.
//! - The [`LanguageModel`] trait that every provider implements.
//! - A CLI-provider adapter ([`cli_adapter`]) for plugging command-line
//!   coding agents into the pipeline without writing a full provider.
//!
//! It has **no** LLM SDK dependencies and no IO-heavy code. Providers
//! targeting Gemini, `OpenAI`, Ollama, etc. live in separate crates.
//!
//! # Porting status
//!
//! | Python module | Status |
//! |---|---|
//! | `core/data.py` | Ported |
//! | `core/types.py` (`FormatType`) | Ported |
//! | `core/exceptions.py` | Ported |
//! | `core/base_model.py` | Ported (as [`model::LanguageModel`]) |
//! | `core/tokenizer.py` | Pending — separate crate |
//! | `core/format_handler.py` | Pending — separate crate |
//! | `core/schema.py` | Pending — separate crate |
//!
//! # Feature flags
//!
//! - `cli-adapter` *(default)* — enables the [`cli_adapter`] module for
//!   wiring command-line coding agents into the [`LanguageModel`] trait.
//!   Opt-out with `default-features = false` if you only want the data
//!   types.

#![forbid(unsafe_code)]

pub mod data;
pub mod error;
pub mod format;
pub mod model;

#[cfg(feature = "cli-adapter")]
pub mod cli_adapter;

// Curated re-exports at crate root so consumers can
// `use langextract_core::{Extraction, LanguageModel, …}` without threading
// module paths everywhere. Explicit, not glob (`M-NO-GLOB-REEXPORTS`).
pub use crate::data::{
    AlignmentStatus, AnnotatedDocument, AttributeMap, AttributeValue, CharInterval, Document,
    DocumentId, ExampleData, Extraction,
};
pub use crate::error::{BoxError, CliRunnerError, InferError, LangExtractError, Result};
pub use crate::format::FormatType;
pub use crate::model::{InferenceParams, LanguageModel, SchemaSupport, ScoredOutput};
