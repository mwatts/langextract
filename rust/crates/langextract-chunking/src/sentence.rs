//! [`SentenceIterator`] — walks a tokenized text sentence by sentence,
//! yielding one [`TokenInterval`] per sentence.
//!
//! Port of `SentenceIterator` in `chunking.py`. Built on top of
//! [`langextract_tokenizer::find_sentence_range`], which we added in
//! increment 2 specifically because the chunker needs it.

use langextract_core::TokenInterval;
use langextract_tokenizer::{TokenizedText, find_sentence_range};

/// Iterator over the sentence intervals of a [`TokenizedText`].
///
/// Borrows the tokenized text. If you want owned iteration, tokenize
/// the text inline and collect into a `Vec<TokenInterval>`.
#[derive(Debug)]
pub struct SentenceIterator<'a> {
    tokenized: &'a TokenizedText,
    cursor: usize,
}

impl<'a> SentenceIterator<'a> {
    /// Construct an iterator starting at the beginning of the tokens.
    #[must_use]
    pub const fn new(tokenized: &'a TokenizedText) -> Self {
        Self {
            tokenized,
            cursor: 0,
        }
    }

    /// Construct an iterator starting at an arbitrary token position.
    ///
    /// # Panics
    ///
    /// Panics if `start > tokenized.tokens.len()`. The Python code
    /// raised `IndexError` — we panic because this is a programming
    /// error, not recoverable user input.
    #[must_use]
    pub fn from_position(tokenized: &'a TokenizedText, start: usize) -> Self {
        assert!(
            start <= tokenized.tokens.len(),
            "SentenceIterator::from_position start {start} > token count {}",
            tokenized.tokens.len()
        );
        Self {
            tokenized,
            cursor: start,
        }
    }

    /// Current token position.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.cursor
    }
}

impl Iterator for SentenceIterator<'_> {
    type Item = TokenInterval;

    fn next(&mut self) -> Option<TokenInterval> {
        if self.cursor >= self.tokenized.tokens.len() {
            return None;
        }
        // find_sentence_range may return a span that starts earlier
        // than our cursor if we're mid-sentence. Clip to [cursor, end).
        let range =
            find_sentence_range(&self.tokenized.text, &self.tokenized.tokens, self.cursor)
                .expect("cursor is in-range; find_sentence_range must succeed");
        let start = self.cursor;
        let end = range.end_index.max(start + 1);
        self.cursor = end;
        Some(TokenInterval::new(start, end))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langextract_tokenizer::tokenize;
    use pretty_assertions::assert_eq;

    #[test]
    fn basic_multi_sentence() {
        // Mirrors `SentenceIterTest::test_basic` from the Python suite.
        let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
        let tokenized = tokenize(text);
        let mut iter = SentenceIterator::new(&tokenized);
        // First sentence: "This is a sentence." → 5 tokens, [0, 5).
        assert_eq!(iter.next(), Some(TokenInterval::new(0, 5)));
        // Second: "This is a longer sentence." → [5, 11).
        assert_eq!(iter.next(), Some(TokenInterval::new(5, 11)));
        // Third: "Mr. Bond\nasks\nwhy?" — Mr. is in the abbreviation
        // whitelist so it doesn't end the sentence; the newline-capital
        // break fires and terminates after "asks". Matches the Python
        // RegexTokenizer + find_sentence_range behaviour.
        let third = iter.next().unwrap();
        assert_eq!(third.start_index, 11);
        assert!(third.end_index > third.start_index);
    }

    #[test]
    fn empty_tokenization_is_empty() {
        let tokenized = tokenize("");
        let mut iter = SentenceIterator::new(&tokenized);
        assert!(iter.next().is_none());
    }
}
