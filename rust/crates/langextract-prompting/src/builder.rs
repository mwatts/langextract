//! Prompt builders — top-level objects that produce one prompt per
//! chunk.
//!
//! Ports `PromptBuilder` and `ContextAwarePromptBuilder` from
//! `langextract/prompting.py`. The Python version uses an inheritance
//! hierarchy with an overridable `build_prompt`; the Rust port uses a
//! trait plus two concrete impls, which is the idiomatic analogue and
//! lets the pipeline hold `&mut dyn PromptBuilder` without caring
//! about the concrete type.

use std::collections::HashMap;

use crate::error::PromptError;
use crate::generator::QAPromptGenerator;

/// Prefix inserted before previous-chunk text in the context window
/// of a [`ContextAwarePromptBuilder`]. Matches the Python constant.
pub const CONTEXT_PREFIX: &str = "[Previous text]: ...";

/// A builder produces one prompt per chunk.
///
/// Implementations may carry mutable state across calls — the
/// [`ContextAwarePromptBuilder`] does exactly that — so the method
/// takes `&mut self`.
pub trait PromptBuilder: Send {
    /// Build a prompt for the given chunk.
    ///
    /// # Arguments
    ///
    /// - `chunk_text` — the text of the current chunk.
    /// - `document_id` — identifier of the source document. Stateful
    ///   builders (context-aware) use this to isolate their state
    ///   per document. Stateless builders ignore it.
    /// - `additional_context` — optional additional context to
    ///   inject into the prompt. `None` or an empty string is
    ///   ignored.
    ///
    /// # Errors
    ///
    /// Returns [`PromptError`] if the underlying renderer fails to
    /// serialize a few-shot example.
    fn build_prompt(
        &mut self,
        chunk_text: &str,
        document_id: &str,
        additional_context: Option<&str>,
    ) -> Result<String, PromptError>;
}

/// Stateless prompt builder — renders every chunk with the same
/// template, ignoring `document_id` and keeping no state between
/// calls. This is the default.
#[derive(Debug, Clone)]
pub struct StatelessPromptBuilder {
    generator: QAPromptGenerator,
}

impl StatelessPromptBuilder {
    /// Wrap a generator.
    #[must_use]
    pub const fn new(generator: QAPromptGenerator) -> Self {
        Self { generator }
    }

    /// Borrow the underlying generator.
    #[must_use]
    pub const fn generator(&self) -> &QAPromptGenerator {
        &self.generator
    }
}

impl PromptBuilder for StatelessPromptBuilder {
    fn build_prompt(
        &mut self,
        chunk_text: &str,
        _document_id: &str,
        additional_context: Option<&str>,
    ) -> Result<String, PromptError> {
        self.generator.render(chunk_text, additional_context)
    }
}

/// Prompt builder that injects text from the previous chunk of the
/// same document as additional context.
///
/// This helps the model resolve cross-chunk coreferences ("she" →
/// "Dr. Sarah Johnson" two chunks back). The window is a trailing
/// byte count from the previous chunk's text, not a token or
/// sentence count — keep it conservative, since it eats into the
/// model's context budget for every chunk after the first.
///
/// State is keyed on `document_id`, so multiple documents can be
/// interleaved through the same builder without context bleeding
/// between them.
#[derive(Debug, Clone)]
pub struct ContextAwarePromptBuilder {
    generator: QAPromptGenerator,
    context_window_chars: Option<usize>,
    prev_chunk_by_doc: HashMap<String, String>,
}

impl ContextAwarePromptBuilder {
    /// Construct a context-aware builder with an optional byte window.
    /// Passing `None` for `context_window_chars` disables cross-chunk
    /// context entirely and behaves like a
    /// [`StatelessPromptBuilder`].
    #[must_use]
    pub fn new(generator: QAPromptGenerator, context_window_chars: Option<usize>) -> Self {
        Self {
            generator,
            context_window_chars,
            prev_chunk_by_doc: HashMap::new(),
        }
    }

    /// The configured window size, if any.
    #[must_use]
    pub const fn context_window_chars(&self) -> Option<usize> {
        self.context_window_chars
    }

    /// Borrow the underlying generator.
    #[must_use]
    pub const fn generator(&self) -> &QAPromptGenerator {
        &self.generator
    }

    /// Drop all per-document context state. Useful when reusing a
    /// long-lived builder for a fresh batch of documents.
    pub fn clear(&mut self) {
        self.prev_chunk_by_doc.clear();
    }

    fn build_effective_context(
        &self,
        document_id: &str,
        additional_context: Option<&str>,
    ) -> Option<String> {
        let mut parts: Vec<String> = Vec::new();

        if let Some(window) = self.context_window_chars {
            if let Some(prev) = self.prev_chunk_by_doc.get(document_id) {
                // Take the trailing `window` bytes of the previous
                // chunk. We use `char_indices` so we never split a
                // UTF-8 code point, even if the window falls
                // mid-character.
                let tail = tail_chars_by_bytes(prev, window);
                parts.push(format!("{CONTEXT_PREFIX}{tail}"));
            }
        }

        if let Some(ctx) = additional_context {
            if !ctx.is_empty() {
                parts.push(ctx.to_owned());
            }
        }

        if parts.is_empty() {
            None
        } else {
            Some(parts.join("\n\n"))
        }
    }

    fn update_state(&mut self, document_id: &str, chunk_text: &str) {
        if self.context_window_chars.is_some() {
            self.prev_chunk_by_doc
                .insert(document_id.to_owned(), chunk_text.to_owned());
        }
    }
}

impl PromptBuilder for ContextAwarePromptBuilder {
    fn build_prompt(
        &mut self,
        chunk_text: &str,
        document_id: &str,
        additional_context: Option<&str>,
    ) -> Result<String, PromptError> {
        let effective = self.build_effective_context(document_id, additional_context);
        let prompt = self
            .generator
            .render(chunk_text, effective.as_deref())?;
        self.update_state(document_id, chunk_text);
        Ok(prompt)
    }
}

/// Return a suffix of `text` that contains at most `window_bytes`
/// trailing bytes, while respecting UTF-8 character boundaries. If the
/// window falls mid-character, we advance forward to the next
/// boundary so the returned slice is always valid UTF-8.
fn tail_chars_by_bytes(text: &str, window_bytes: usize) -> &str {
    if text.len() <= window_bytes {
        return text;
    }
    let start = text.len() - window_bytes;
    // Advance to the next char boundary.
    let mut boundary = start;
    while boundary < text.len() && !text.is_char_boundary(boundary) {
        boundary += 1;
    }
    &text[boundary..]
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn tail_respects_char_boundary() {
        // "naïve" in UTF-8 is 6 bytes ('n', 'a', 'ï'=2 bytes, 'v', 'e').
        // Requesting the last 5 bytes would land mid-char; we should
        // round forward.
        let s = "naïve";
        let out = tail_chars_by_bytes(s, 5);
        assert!(
            out.is_char_boundary(0),
            "returned slice must be valid UTF-8 at start"
        );
        assert!(out.chars().count() <= s.chars().count());
    }

    #[test]
    fn tail_shorter_than_window_returns_whole() {
        assert_eq!(tail_chars_by_bytes("abc", 100), "abc");
    }
}
