//! JSON/YAML output formatting and LLM output parsing for the
//! langextract Rust port.
//!
//! Port of `langextract/core/format_handler.py`. The crate centralises
//! everything the pipeline needs to turn raw LLM output into structured
//! extraction records and to build the few-shot prompt examples that
//! tell the model what shape to return.
//!
//! # Features
//!
//! - [`FormatHandler`] with Python-parity defaults (JSON + wrapper +
//!   fences + `_attributes` suffix) and a builder for overrides.
//! - [`FormatHandler::parse_output`] — fenced-block aware parser with
//!   wrapper-key extraction, top-level list fallback, strict mode,
//!   and automatic `<think>...</think>` stripping for reasoning
//!   models (DeepSeek-R1, `QwQ`) that interleave chain-of-thought with
//!   the actual JSON/YAML.
//! - [`FormatHandler::format_extraction_example`] — emit JSON or YAML
//!   for a prompt's few-shot examples, optionally wrapped in a code
//!   fence with the correct language tag.
//! - [`fence::find_fenced_blocks`] — standalone fence scanner you can
//!   use if you only want the fence-detection half, e.g. to write a
//!   custom parser that still reuses the language-tag filter.
//! - [`fence::strip_think_tags`] — standalone helper for models that
//!   always emit reasoning and need it stripped before any parsing.
//!
//! # What is not ported
//!
//! The `from_resolver_params` / `from_kwargs` legacy shims in Python
//! exist to migrate callers from the pre-1.0 resolver API. The Rust
//! port has no such legacy to maintain; the builder plus direct
//! constructors cover every reasonable configuration in a few lines.

#![forbid(unsafe_code)]

pub mod error;
pub mod fence;
pub mod handler;

pub use crate::error::FormatError;
pub use crate::handler::{FormatHandler, FormatHandlerBuilder, ParsedOutput, ParsedRecord};
