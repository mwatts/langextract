//! Hand-rolled regex-equivalent tokenizer.
//!
//! This is the Rust analogue of Python's `RegexTokenizer` in
//! `core/tokenizer.py`. The Python implementation uses the pattern
//!
//! ```text
//! [^\W\d_]+ | \d+ | ([^\w\s]|_)\1*
//! ```
//!
//! which relies on a backreference (`\1`) to group runs of identical
//! symbols. Rust's `regex` crate does not support backreferences, and
//! pulling in `fancy-regex` or `onig` just for this one pattern would be
//! wasteful: the logic is simple enough to express directly as a
//! character-level scanner, which is what this module does.
//!
//! The classification rules are:
//!
//! - **Whitespace** (Unicode `White_Space`) is skipped but remembered — if
//!   a gap between two non-whitespace runs contains `\n` or `\r`, the next
//!   token is marked [`first_token_after_newline`](crate::Token::first_token_after_newline).
//! - **Letters** (Unicode alphabetic, excluding `_`) form a
//!   [`Word`](crate::TokenType::Word) token. Consecutive letters are merged.
//! - **ASCII digits** (`0-9`) form a [`Number`](crate::TokenType::Number)
//!   token. Non-ASCII numeric characters are deferred to a future
//!   `UnicodeTokenizer` crate.
//! - **Anything else** forms a
//!   [`Punctuation`](crate::TokenType::Punctuation) token. **Runs of the
//!   identical character are merged** (`"!!!"` is one token, `"!?"` is
//!   two) — this matches the `(\1*)` semantics of the Python pattern.
//!
//! Indices stored in each token are **byte offsets** into the source
//! `&str`; see [`crate::types`] for rationale.

use core::iter::Peekable;
use core::str::CharIndices;

use langextract_core::CharInterval;

use crate::types::{Token, TokenType, TokenizedText};
use crate::Tokenizer;

/// Default tokenizer. Fast, English-friendly, no allocations per token.
#[derive(Debug, Default, Clone, Copy)]
pub struct RegexTokenizer;

impl RegexTokenizer {
    /// Construct a new tokenizer. Const for use in static initialisers.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

impl Tokenizer for RegexTokenizer {
    fn tokenize(&self, text: &str) -> TokenizedText {
        let mut out = TokenizedText::new(text);
        let bytes = text.as_bytes();
        let mut chars = text.char_indices().peekable();
        let mut prev_end: usize = 0;

        while let Some(&(start, c)) = chars.peek() {
            if c.is_whitespace() {
                chars.next();
                continue;
            }

            // Detect whether the gap since the previous token contained a
            // newline. The gap is `bytes[prev_end..start]`. Scan raw bytes
            // because `\n`/`\r` are ASCII and contained in their own byte.
            let gap_has_newline = bytes[prev_end..start]
                .iter()
                .any(|&b| b == b'\n' || b == b'\r');

            let (token_type, end) = if c.is_ascii_digit() {
                (
                    TokenType::Number,
                    consume_while(&mut chars, start, c, |nc| nc.is_ascii_digit()),
                )
            } else if is_letter(c) {
                (
                    TokenType::Word,
                    consume_while(&mut chars, start, c, is_letter),
                )
            } else {
                // Symbol / underscore: group identical characters only.
                (
                    TokenType::Punctuation,
                    consume_while(&mut chars, start, c, |nc| nc == c),
                )
            };

            let first_after_newline = gap_has_newline && !out.tokens.is_empty();

            out.tokens.push(Token {
                index: out.tokens.len(),
                token_type,
                char_interval: CharInterval::new(start, end),
                first_token_after_newline: first_after_newline,
            });
            prev_end = end;
        }

        out
    }
}

/// Consume characters while `pred` holds. Caller has already *peeked*
/// `first_c` at byte offset `first_start` but not yet advanced the
/// iterator. Returns the byte offset one past the end of the run.
fn consume_while<F>(
    chars: &mut Peekable<CharIndices<'_>>,
    first_start: usize,
    first_c: char,
    pred: F,
) -> usize
where
    F: Fn(char) -> bool,
{
    // Consume the first char (we already know it qualifies).
    chars.next();
    let mut end = first_start + first_c.len_utf8();
    while let Some(&(pos, nc)) = chars.peek() {
        if pred(nc) {
            chars.next();
            end = pos + nc.len_utf8();
        } else {
            break;
        }
    }
    end
}

/// `true` if `c` is a "letter" in the sense of Python's `[^\W\d_]`:
/// alphabetic, not a digit, not underscore.
fn is_letter(c: char) -> bool {
    c.is_alphabetic() && !c.is_ascii_digit() && c != '_'
}
