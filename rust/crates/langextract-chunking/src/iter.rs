//! [`ChunkIterator`] — the core chunking algorithm.
//!
//! Port of `ChunkIterator` in `langextract/chunking.py`. The algorithm
//! has three branches, spelled out in the Python source and reproduced
//! here so the port is self-documenting:
//!
//! **(A) Long sentence.** If a single sentence is longer than
//! `max_char_buffer`, it is broken into chunks that fit the buffer.
//! The break point prefers the most recent newline the chunk has
//! crossed, so source-line structure is respected where possible.
//!
//! **(B) Single oversized token.** If even the first token of a
//! sentence exceeds the buffer, that token comprises an entire chunk
//! on its own. No other tokens share the chunk.
//!
//! **(C) Multiple whole sentences.** If several consecutive whole
//! sentences fit within the buffer together, they are packed into a
//! single chunk rather than being emitted individually.
//!
//! The `broken_sentence` flag on the iterator tracks whether the last
//! emission truncated a sentence mid-way, so the next call knows not
//! to try extending the chunk with subsequent whole sentences (it
//! should finish the current sentence first).

use langextract_core::{Document, DocumentId, TokenInterval};
use langextract_tokenizer::{RegexTokenizer, TokenizedText, Tokenizer};

use crate::chunk::{TextChunk, get_char_interval};
use crate::error::ChunkingError;
use crate::sentence::SentenceIterator;

/// Iterator over [`TextChunk`]s of a tokenized document.
///
/// Construct via [`ChunkIterator::new`] (uses the default
/// [`RegexTokenizer`]) or [`ChunkIterator::from_tokenized`] if you
/// already have a [`TokenizedText`].
///
/// Each call to [`Iterator::next`] returns the next chunk, fully
/// materialized (char interval, token interval, owned text).
#[derive(Debug)]
pub struct ChunkIterator {
    tokenized: TokenizedText,
    max_char_buffer: usize,
    cursor: usize,
    broken_sentence: bool,
    document_id: Option<DocumentId>,
    additional_context: Option<String>,
}

impl ChunkIterator {
    /// Construct a chunker over raw text using the default
    /// [`RegexTokenizer`].
    ///
    /// # Errors
    ///
    /// Returns [`ChunkingError::InvalidBufferSize`] if
    /// `max_char_buffer == 0`.
    pub fn new(text: impl Into<String>, max_char_buffer: usize) -> Result<Self, ChunkingError> {
        let tokenized = RegexTokenizer::new().tokenize(&text.into());
        Self::from_tokenized(tokenized, max_char_buffer)
    }

    /// Construct a chunker using a custom tokenizer.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkingError::InvalidBufferSize`] if
    /// `max_char_buffer == 0`.
    pub fn with_tokenizer<T: Tokenizer>(
        text: &str,
        max_char_buffer: usize,
        tokenizer: &T,
    ) -> Result<Self, ChunkingError> {
        let result = tokenizer.tokenize(text);
        Self::from_tokenized(result, max_char_buffer)
    }

    /// Construct a chunker from an already-tokenized document.
    ///
    /// # Errors
    ///
    /// Returns [`ChunkingError::InvalidBufferSize`] if
    /// `max_char_buffer == 0`.
    pub fn from_tokenized(
        tokenized: TokenizedText,
        max_char_buffer: usize,
    ) -> Result<Self, ChunkingError> {
        if max_char_buffer == 0 {
            return Err(ChunkingError::InvalidBufferSize(0));
        }
        Ok(Self {
            tokenized,
            max_char_buffer,
            cursor: 0,
            broken_sentence: false,
            document_id: None,
            additional_context: None,
        })
    }

    /// Attach a source [`Document`] whose metadata (`document_id`,
    /// `additional_context`) should be propagated onto every emitted
    /// chunk. Mutates the iterator in place and returns it for
    /// chaining.
    #[must_use]
    pub fn with_document(mut self, document: &Document) -> Self {
        self.document_id.clone_from(&document.document_id);
        self.additional_context
            .clone_from(&document.additional_context);
        self
    }

    /// Set the document ID directly (useful if you don't have a
    /// full `Document` but still want the ID on each chunk).
    #[must_use]
    pub fn with_document_id(mut self, id: DocumentId) -> Self {
        self.document_id = Some(id);
        self
    }

    // ---------- internal helpers ----------

    fn tokens_exceed_buffer(&self, interval: TokenInterval) -> bool {
        let char_interval =
            get_char_interval(&self.tokenized, interval).expect("valid interval");
        (char_interval.end - char_interval.start) > self.max_char_buffer
    }

    fn make_chunk(&self, interval: TokenInterval) -> TextChunk {
        let char_interval =
            get_char_interval(&self.tokenized, interval).expect("valid interval");
        let text = self.tokenized.text[char_interval.start..char_interval.end].to_owned();
        TextChunk {
            token_interval: interval,
            char_interval,
            text,
            document_id: self.document_id.clone(),
            additional_context: self.additional_context.clone(),
        }
    }

    fn next_sentence(&self, from: usize) -> Option<TokenInterval> {
        SentenceIterator::from_position(&self.tokenized, from).next()
    }
}

impl Iterator for ChunkIterator {
    type Item = TextChunk;

    fn next(&mut self) -> Option<TextChunk> {
        let sentence = self.next_sentence(self.cursor)?;

        // Branch (B): If the first token of this sentence alone
        // exceeds the buffer, emit it as its own chunk.
        let mut curr_chunk = TokenInterval::new(sentence.start_index, sentence.start_index + 1);
        if self.tokens_exceed_buffer(curr_chunk) {
            self.cursor = sentence.start_index + 1;
            self.broken_sentence = curr_chunk.end_index < sentence.end_index;
            return Some(self.make_chunk(curr_chunk));
        }

        // Branch (A): Expand the chunk token-by-token within the
        // current sentence. Track the most recent newline so we can
        // truncate to a line boundary when the buffer overflows.
        let mut start_of_new_line: Option<usize> = None;
        for token_index in curr_chunk.start_index..sentence.end_index {
            if self.tokenized.tokens[token_index].first_token_after_newline {
                start_of_new_line = Some(token_index);
            }
            let test_chunk = TokenInterval::new(curr_chunk.start_index, token_index + 1);
            if self.tokens_exceed_buffer(test_chunk) {
                // Truncate to the most recent newline inside this
                // chunk, if one exists and it's strictly after the
                // chunk start (so the resulting interval is
                // non-empty).
                if let Some(nl) = start_of_new_line {
                    if nl > 0 && nl > curr_chunk.start_index {
                        curr_chunk = TokenInterval::new(curr_chunk.start_index, nl);
                    }
                }
                self.cursor = curr_chunk.end_index;
                self.broken_sentence = true;
                return Some(self.make_chunk(curr_chunk));
            }
            curr_chunk = test_chunk;
        }

        // Whole sentence fit. Branch (C): pack subsequent whole
        // sentences into the same chunk as long as the combined span
        // still fits the buffer. Skip this if we're still recovering
        // from a previous mid-sentence break.
        if self.broken_sentence {
            self.broken_sentence = false;
        } else {
            let mut probe = self.next_sentence(curr_chunk.end_index);
            while let Some(next_sentence) = probe {
                let test_chunk =
                    TokenInterval::new(curr_chunk.start_index, next_sentence.end_index);
                if self.tokens_exceed_buffer(test_chunk) {
                    break;
                }
                curr_chunk = test_chunk;
                probe = self.next_sentence(curr_chunk.end_index);
            }
        }

        self.cursor = curr_chunk.end_index;
        Some(self.make_chunk(curr_chunk))
    }
}

/// Batch chunks into fixed-size groups.
///
/// Port of `make_batches_of_textchunk`. Useful when a provider accepts
/// a batch of prompts per call — you chunk the document, then batch
/// the chunks up to the provider's batch limit. Trailing partial
/// batches are yielded as-is.
///
/// # Panics
///
/// Panics if `batch_size == 0`.
pub fn batch_chunks<I>(iter: I, batch_size: usize) -> impl Iterator<Item = Vec<TextChunk>>
where
    I: Iterator<Item = TextChunk>,
{
    assert!(batch_size > 0, "batch_size must be >= 1");
    BatchIter {
        iter,
        batch_size,
        done: false,
    }
}

struct BatchIter<I: Iterator<Item = TextChunk>> {
    iter: I,
    batch_size: usize,
    done: bool,
}

impl<I: Iterator<Item = TextChunk>> Iterator for BatchIter<I> {
    type Item = Vec<TextChunk>;

    fn next(&mut self) -> Option<Vec<TextChunk>> {
        if self.done {
            return None;
        }
        let mut batch = Vec::with_capacity(self.batch_size);
        for _ in 0..self.batch_size {
            if let Some(c) = self.iter.next() {
                batch.push(c);
            } else {
                self.done = true;
                break;
            }
        }
        if batch.is_empty() { None } else { Some(batch) }
    }
}

// Make BatchIter itself Debug so `missing_debug_implementations` is
// satisfied — required by crate-level lint config.
impl<I: Iterator<Item = TextChunk>> std::fmt::Debug for BatchIter<I> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BatchIter")
            .field("batch_size", &self.batch_size)
            .field("done", &self.done)
            .finish_non_exhaustive()
    }
}
