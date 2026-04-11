//! Token-level normalization helpers.
//!
//! Port of the two private helpers at the bottom of
//! `langextract/resolver.py`:
//!
//! - `_tokenize_with_lowercase` — tokenize text and yield lowercased token
//!   strings.
//! - `_normalize_token` — lowercase + strip a trailing single `s` for
//!   tokens longer than 3 characters (light pluralisation stemming). Does
//!   not strip `ss`, so `"class"` stays `"class"`.
//!
//! The two-stage design (raw lowercase for exact matching, normalized for
//! fuzzy matching) mirrors the Python resolver so that matches are
//! computed against the same canonical form.

use langextract_tokenizer::{RegexTokenizer, Tokenizer, TokenizedText};

/// Tokenize `text` with the given tokenizer and return its tokens as
/// lowercase strings, in order. Whitespace and empty tokens are dropped
/// by construction (the tokenizer doesn't emit them).
#[must_use]
pub fn lowercase_tokens<T: Tokenizer>(text: &str, tokenizer: &T) -> Vec<String> {
    let result = tokenizer.tokenize(text);
    lowercase_tokens_from(&result)
}

/// Like [`lowercase_tokens`] but operates on an already-tokenized text.
/// Useful when the caller already holds a [`TokenizedText`] and wants to
/// avoid re-tokenizing.
#[must_use]
pub fn lowercase_tokens_from(tokenized: &TokenizedText) -> Vec<String> {
    tokenized
        .tokens
        .iter()
        .filter_map(|tok| tokenized.token_text(tok).map(str::to_lowercase))
        .collect()
}

/// Convenience: tokenize using the default [`RegexTokenizer`].
#[must_use]
pub fn default_lowercase_tokens(text: &str) -> Vec<String> {
    lowercase_tokens(text, &RegexTokenizer::new())
}

/// Lowercase a token and apply the same light pluralisation stemming as
/// Python's `_normalize_token`: strip a single trailing `s` for tokens
/// longer than 3 characters, but not `ss`.
#[must_use]
pub fn normalize_token(token: &str) -> String {
    let lower = token.to_lowercase();
    if lower.chars().count() > 3 && lower.ends_with('s') && !lower.ends_with("ss") {
        // Pop one char from the end. `s` is ASCII so byte-slicing is
        // safe, but we go via char_indices to be Unicode-robust.
        if let Some((idx, _)) = lower.char_indices().next_back() {
            return lower[..idx].to_owned();
        }
    }
    lower
}

/// Normalize an entire slice of tokens.
#[must_use]
pub fn normalize_tokens(tokens: &[String]) -> Vec<String> {
    tokens.iter().map(|t| normalize_token(t)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn normalize_strips_plural_s_over_three_chars() {
        assert_eq!(normalize_token("problems"), "problem");
        assert_eq!(normalize_token("heart"), "heart");
    }

    #[test]
    fn normalize_preserves_ss_ending() {
        assert_eq!(normalize_token("class"), "class");
        assert_eq!(normalize_token("ASSESS"), "assess");
    }

    #[test]
    fn normalize_preserves_short_tokens() {
        // 3-char and shorter tokens are not stemmed, even if they end in s.
        assert_eq!(normalize_token("bus"), "bus");
        assert_eq!(normalize_token("is"), "is");
    }

    #[test]
    fn normalize_lowercases() {
        assert_eq!(normalize_token("Heart"), "heart");
        assert_eq!(normalize_token("NAPROSYN"), "naprosyn");
    }

    #[test]
    fn default_lowercase_tokens_basic() {
        let toks = default_lowercase_tokens("Patient has Heart Problems.");
        assert_eq!(
            toks,
            vec!["patient", "has", "heart", "problems", "."]
        );
    }
}
