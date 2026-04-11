//! Split long documents into sentence-aware chunks that fit a
//! language model's context window.
//!
//! Port of `langextract/chunking.py`. The pipeline uses this to break
//! up source documents before handing them to a
//! [`LanguageModel`](langextract_core::LanguageModel) with a bounded
//! context — every chunk that comes out of [`ChunkIterator`] fits
//! within a caller-specified byte budget, and the tokenizer's
//! sentence-boundary detector keeps chunks from cutting across a
//! sentence unless the sentence itself is longer than the budget.
//!
//! # Quick start
//!
//! ```no_run
//! use langextract_chunking::ChunkIterator;
//!
//! let chunker = ChunkIterator::new(long_document(), 1500).unwrap();
//! for chunk in chunker {
//!     // chunk.text is the chunk body; chunk.token_interval /
//!     // chunk.char_interval tell you where it came from in the
//!     // source, which is what the aligner needs later to restore
//!     // absolute offsets.
//!     infer_and_extract(&chunk);
//! }
//! # fn long_document() -> String { String::new() }
//! # fn infer_and_extract(_: &langextract_chunking::TextChunk) {}
//! ```
//!
//! # Choosing a `max_char_buffer`
//!
//! Pick a conservative value below your model's real context limit.
//! Remember that the prompt itself — the instructions and few-shot
//! examples — will also eat into that context. For a CLI provider
//! (Claude Code, aider, gemini-cli, …), start high (≥ 4000) because
//! each invocation is expensive; for a cheap API you can go lower to
//! improve parallelism.
//!
//! # The three chunking rules
//!
//! Exactly as in the Python source, see [`ChunkIterator`] for the
//! inline documentation, but briefly:
//!
//! 1. Multiple whole sentences are packed together when they fit.
//! 2. A single long sentence is broken at the most recent newline
//!    inside the buffer, falling back to a token boundary if no
//!    newline is present.
//! 3. A single token that exceeds the buffer is emitted alone.

#![forbid(unsafe_code)]

pub mod chunk;
pub mod error;
pub mod iter;
pub mod sentence;

pub use crate::chunk::{
    TextChunk, create_token_interval, get_char_interval, get_token_interval_text,
};
pub use crate::error::ChunkingError;
pub use crate::iter::{ChunkIterator, batch_chunks};
pub use crate::sentence::SentenceIterator;
