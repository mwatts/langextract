//! Sentence boundary detection.
//!
//! Port of `tokens_text`, `find_sentence_range`, and the private
//! `_is_end_of_sentence_token` / `_is_sentence_break_after_newline`
//! helpers in `core/tokenizer.py`.
//!
//! The algorithm walks tokens from a starting index and returns a
//! [`TokenInterval`] covering the first detected sentence. A token ends a
//! sentence when it is a punctuation token whose text contains an
//! end-of-sentence marker (`.`, `?`, `!`, or the Chinese/Japanese/Hindi
//! equivalents `。！？।`) and is not part of a known abbreviation
//! (`Dr.`, `Mrs.`, …). Trailing closing punctuation (`"`, `)`, `]`, etc.)
//! is folded into the returned interval so that `He said "Hi."` ends
//! after the closing quote, not after the period.

use std::collections::HashSet;
use std::hash::BuildHasher;
use std::sync::OnceLock;

use crate::error::TokenizerError;
use crate::types::{Token, TokenInterval, TokenType, TokenizedText};

/// Default abbreviations that should not count as sentence terminators.
///
/// Mirrors `_KNOWN_ABBREVIATIONS` in `core/tokenizer.py`.
pub const DEFAULT_ABBREVIATIONS: &[&str] = &["Mr.", "Mrs.", "Ms.", "Dr.", "Prof.", "St."];

/// Characters that count as end-of-sentence markers.
///
/// ASCII `.?!` plus CJK and Devanagari equivalents.
const END_OF_SENTENCE_CHARS: &[char] = &['.', '?', '!', '。', '！', '？', '\u{0964}'];

/// Characters that are treated as closing punctuation and folded into a
/// sentence after its terminating marker.
const CLOSING_PUNCTUATION: &[char] =
    &['"', '\'', '\u{201D}', '\u{2019}', '\u{00BB}', ')', ']', '}'];

fn default_abbreviation_set() -> &'static HashSet<&'static str> {
    static SET: OnceLock<HashSet<&'static str>> = OnceLock::new();
    SET.get_or_init(|| DEFAULT_ABBREVIATIONS.iter().copied().collect())
}

/// Reconstruct the substring of the source text spanned by a token interval.
///
/// Port of `tokens_text`. An empty interval (`start == end`) returns `""`.
///
/// # Errors
///
/// Returns [`TokenizerError::InvalidTokenInterval`] if the interval is
/// out of bounds or has `start > end`.
pub fn tokens_text(
    tokenized: &TokenizedText,
    interval: TokenInterval,
) -> Result<&str, TokenizerError> {
    if interval.is_empty() {
        return Ok("");
    }
    let total = tokenized.tokens.len();
    if interval.end_index > total || interval.start_index > interval.end_index {
        return Err(TokenizerError::InvalidTokenInterval {
            start: interval.start_index,
            end: interval.end_index,
            total,
        });
    }
    let start_byte = tokenized.tokens[interval.start_index].char_interval.start;
    let end_byte = tokenized.tokens[interval.end_index - 1].char_interval.end;
    tokenized
        .text
        .get(start_byte..end_byte)
        .ok_or(TokenizerError::InvalidTokenInterval {
            start: interval.start_index,
            end: interval.end_index,
            total,
        })
}

/// Find a sentence interval starting at `start_index`.
///
/// Port of `find_sentence_range`. Convenience overload that uses the
/// default abbreviation set; callers who want to customise should use
/// [`find_sentence_range_with`].
///
/// # Errors
///
/// Returns [`TokenizerError::SentenceStartOutOfRange`] if
/// `start_index >= tokens.len()` (unless `tokens` is empty, in which case
/// an empty interval `[0, 0)` is returned).
pub fn find_sentence_range(
    text: &str,
    tokens: &[Token],
    start_index: usize,
) -> Result<TokenInterval, TokenizerError> {
    find_sentence_range_with(text, tokens, start_index, default_abbreviation_set())
}

/// Find a sentence interval starting at `start_index`, using a custom
/// abbreviation whitelist.
///
/// # Errors
///
/// Returns [`TokenizerError::SentenceStartOutOfRange`] if
/// `start_index >= tokens.len()` (unless `tokens` is empty, in which case
/// an empty interval `[0, 0)` is returned).
pub fn find_sentence_range_with<S: BuildHasher>(
    text: &str,
    tokens: &[Token],
    start_index: usize,
    abbreviations: &HashSet<&str, S>,
) -> Result<TokenInterval, TokenizerError> {
    if tokens.is_empty() {
        return Ok(TokenInterval::new(0, 0));
    }
    if start_index >= tokens.len() {
        return Err(TokenizerError::SentenceStartOutOfRange {
            start: start_index,
            total: tokens.len(),
        });
    }

    let mut i = start_index;
    while i < tokens.len() {
        if tokens[i].token_type == TokenType::Punctuation
            && is_end_of_sentence_token(text, tokens, i, abbreviations)
        {
            // Absorb trailing closing punctuation.
            let mut end_index = i + 1;
            while end_index < tokens.len() {
                let tok = &tokens[end_index];
                if tok.token_type == TokenType::Punctuation
                    && token_text_is_closing(text, tok)
                {
                    end_index += 1;
                } else {
                    break;
                }
            }
            return Ok(TokenInterval::new(start_index, end_index));
        }

        if is_sentence_break_after_newline(text, tokens, i) {
            return Ok(TokenInterval::new(start_index, i + 1));
        }

        i += 1;
    }

    Ok(TokenInterval::new(start_index, tokens.len()))
}

// ---------- helpers ----------

fn token_str<'a>(text: &'a str, tok: &Token) -> &'a str {
    &text[tok.char_interval.start..tok.char_interval.end]
}

/// `true` if the punctuation token at `idx` should terminate a sentence.
///
/// Matches `_END_OF_SENTENCE_PATTERN` against the token text, then
/// checks whether concatenating with the preceding token produces a known
/// abbreviation (in which case it is **not** a sentence ender).
fn is_end_of_sentence_token<S: BuildHasher>(
    text: &str,
    tokens: &[Token],
    idx: usize,
    abbreviations: &HashSet<&str, S>,
) -> bool {
    let token_text = token_str(text, &tokens[idx]);
    if !token_text_matches_eos(token_text) {
        return false;
    }
    if idx > 0 {
        let prev_text = token_str(text, &tokens[idx - 1]);
        // Python does `f"{prev}{current}"` — a single allocation per check.
        let mut combined = String::with_capacity(prev_text.len() + token_text.len());
        combined.push_str(prev_text);
        combined.push_str(token_text);
        if abbreviations.contains(combined.as_str()) {
            return false;
        }
    }
    true
}

/// Python pattern `[.?!。！？।][closers]*$`. Because
/// [`RegexTokenizer`](crate::RegexTokenizer) emits punctuation tokens as
/// either a single symbol or a run of identical symbols, a token qualifies
/// iff its last character is in [`END_OF_SENTENCE_CHARS`]. The optional
/// trailing closers are absorbed as subsequent tokens by
/// [`find_sentence_range_with`].
fn token_text_matches_eos(token_text: &str) -> bool {
    token_text
        .chars()
        .next_back()
        .is_some_and(|c| END_OF_SENTENCE_CHARS.contains(&c))
}

fn token_text_is_closing(text: &str, tok: &Token) -> bool {
    let s = token_str(text, tok);
    // Token is a run of identical chars; check the first.
    s.chars()
        .next()
        .is_some_and(|c| CLOSING_PUNCTUATION.contains(&c))
}

/// `true` if the *next* token starts after a newline and begins with an
/// uppercase character. Mirrors `_is_sentence_break_after_newline`.
fn is_sentence_break_after_newline(text: &str, tokens: &[Token], idx: usize) -> bool {
    let Some(next) = tokens.get(idx + 1) else {
        return false;
    };
    if !next.first_token_after_newline {
        return false;
    }
    let next_text = token_str(text, next);
    next_text.chars().next().is_some_and(|c| !c.is_lowercase())
}
