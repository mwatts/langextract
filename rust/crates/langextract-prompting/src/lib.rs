//! Few-shot prompt construction for the Rust port of langextract.
//!
//! Port of `langextract/prompting.py`. This crate wraps a
//! [`PromptTemplateStructured`] (the user-authored description plus
//! few-shot examples) and a
//! [`FormatHandler`](langextract_format::FormatHandler) (the output
//! format rules from the previous increment) into a single
//! [`QAPromptGenerator`] that can render a complete prompt for any
//! input chunk. Two builder flavours sit on top:
//!
//! - [`StatelessPromptBuilder`] — renders every chunk independently.
//!   Use this when chunks are self-contained or when chunking has
//!   been set up to always break on sentence boundaries.
//! - [`ContextAwarePromptBuilder`] — carries a trailing slice of the
//!   previous chunk forward into the next prompt, keyed by
//!   `document_id` so multiple documents can be interleaved without
//!   cross-contamination. Use this when the model needs to resolve
//!   coreferences across chunk boundaries.
//!
//! # Pipeline wiring
//!
//! The full chunked-extraction loop (using everything the Rust port
//! provides so far) now looks like the following — this pulls in
//! crates the prompting crate doesn't itself depend on, so the
//! rustdoc block is marked `ignore` to avoid forcing a circular dev
//! dependency graph:
//!
//! ```ignore
//! use langextract_aligner::{align_extraction_groups_with, AlignmentOptions};
//! use langextract_chunking::ChunkIterator;
//! use langextract_core::{ATTRIBUTE_SUFFIX, Extraction, InferenceParams, LanguageModel};
//! use langextract_format::{extract_ordered_extractions, FormatHandler};
//! use langextract_prompting::{PromptBuilder, QAPromptGenerator, StatelessPromptBuilder};
//! use langextract_tokenizer::RegexTokenizer;
//!
//! async fn run(
//!     model: &dyn LanguageModel,
//!     template: langextract_prompting::PromptTemplateStructured,
//!     source: &str,
//! ) -> Result<Vec<Extraction>, Box<dyn std::error::Error + Send + Sync>> {
//!     let handler = FormatHandler::new();
//!     let generator = QAPromptGenerator::new(template, handler.clone());
//!     let mut builder = StatelessPromptBuilder::new(generator);
//!     let tokenizer = RegexTokenizer::new();
//!     let mut grounded = Vec::new();
//!     for chunk in ChunkIterator::new(source.to_owned(), 1500)? {
//!         let prompt = builder.build_prompt(&chunk.text, "doc", None)?;
//!         let outputs = model.infer(&[prompt], &InferenceParams::default()).await?;
//!         let raw = &outputs[0][0].output;
//!         let records = handler.parse_output(raw, None)?;
//!         let extractions =
//!             extract_ordered_extractions(&records, ATTRIBUTE_SUFFIX, Some("_index"))?;
//!         let chunk_grounded = align_extraction_groups_with(
//!             vec![extractions],
//!             &chunk.text,
//!             &AlignmentOptions {
//!                 token_offset: chunk.token_interval.start_index,
//!                 char_offset: chunk.char_interval.start,
//!                 ..Default::default()
//!             },
//!             &tokenizer,
//!         )?;
//!         grounded.extend(chunk_grounded.into_iter().flatten());
//!     }
//!     Ok(grounded)
//! }
//! ```
//!
//! The only piece still missing is a top-level `extract()` facade —
//! see `CLAUDE.md` for the remaining work.

#![forbid(unsafe_code)]

pub mod builder;
pub mod error;
pub mod generator;
pub mod template;

pub use crate::builder::{
    CONTEXT_PREFIX, ContextAwarePromptBuilder, PromptBuilder, StatelessPromptBuilder,
};
pub use crate::error::PromptError;
pub use crate::generator::{
    DEFAULT_ANSWER_PREFIX, DEFAULT_EXAMPLES_HEADING, DEFAULT_QUESTION_PREFIX, QAPromptGenerator,
    QAPromptGeneratorBuilder,
};
pub use crate::template::{PromptTemplateStructured, read_prompt_template_from_file};
