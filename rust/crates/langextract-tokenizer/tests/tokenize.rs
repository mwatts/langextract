//! Integration tests mirroring selected parameterized cases from
//! `tests/tokenizer_test.py::TokenizerTest`.

use langextract_tokenizer::{
    find_sentence_range, tokenize, tokens_text, RegexTokenizer, Token, TokenInterval, TokenType,
    Tokenizer, TokenizerError,
};
use pretty_assertions::assert_eq;

/// Shorthand: just collect the (`token_type`, `first_token_after_newline`)
/// pairs from a tokenization, which is what the Python parameterized
/// tests effectively compare.
fn shape(tokens: &[Token]) -> Vec<(TokenType, bool)> {
    tokens
        .iter()
        .map(|t| (t.token_type, t.first_token_after_newline))
        .collect()
}

#[test]
fn basic_text() {
    use TokenType::{Punctuation, Word};
    // "Hello, world!" → WORD, PUNCT, WORD, PUNCT
    let tokenized = tokenize("Hello, world!");
    assert_eq!(
        shape(&tokenized.tokens),
        vec![
            (Word, false),
            (Punctuation, false),
            (Word, false),
            (Punctuation, false),
        ]
    );
}

#[test]
fn multiple_spaces_and_numbers() {
    use TokenType::{Number, Punctuation, Word};
    // "Age:   25\nWeight=70kg." → WORD, PUNCT, NUMBER, WORD(nl), PUNCT, NUMBER, WORD, PUNCT
    let tokenized = tokenize("Age:   25\nWeight=70kg.");
    assert_eq!(
        shape(&tokenized.tokens),
        vec![
            (Word, false),
            (Punctuation, false),
            (Number, false),
            (Word, true),
            (Punctuation, false),
            (Number, false),
            (Word, false),
            (Punctuation, false),
        ]
    );
}

#[test]
fn multi_line_input() {
    use TokenType::{Number, Word};
    // "Line1\nLine2\nLine3" → WORD, NUMBER, WORD(nl), NUMBER, WORD(nl), NUMBER
    let tokenized = tokenize("Line1\nLine2\nLine3");
    assert_eq!(
        shape(&tokenized.tokens),
        vec![
            (Word, false),
            (Number, false),
            (Word, true),
            (Number, false),
            (Word, true),
            (Number, false),
        ]
    );
}

#[test]
fn only_symbols_groups_identical_splits_mixed() {
    // "!!!@#   $$$%" → !!!, @, #, $$$, %
    let tokenized = tokenize("!!!@#   $$$%");
    assert_eq!(tokenized.tokens.len(), 5);
    assert_eq!(tokenized.token_text(&tokenized.tokens[0]), Some("!!!"));
    assert_eq!(tokenized.token_text(&tokenized.tokens[1]), Some("@"));
    assert_eq!(tokenized.token_text(&tokenized.tokens[2]), Some("#"));
    assert_eq!(tokenized.token_text(&tokenized.tokens[3]), Some("$$$"));
    assert_eq!(tokenized.token_text(&tokenized.tokens[4]), Some("%"));
}

#[test]
fn empty_string() {
    let tokenized = tokenize("");
    assert!(tokenized.is_empty());
}

#[test]
fn non_ascii_single_word() {
    let tokenized = tokenize("café");
    assert_eq!(tokenized.tokens.len(), 1);
    assert_eq!(tokenized.tokens[0].token_type, TokenType::Word);
    // Byte-offset slicing must round-trip.
    assert_eq!(tokenized.token_text(&tokenized.tokens[0]), Some("café"));
}

#[test]
fn mixed_punctuation_splits() {
    // "?!" → ? and !
    let tokenized = tokenize("?!");
    assert_eq!(tokenized.tokens.len(), 2);
    assert_eq!(tokenized.token_text(&tokenized.tokens[0]), Some("?"));
    assert_eq!(tokenized.token_text(&tokenized.tokens[1]), Some("!"));
}

#[test]
fn underscore_splits_word() {
    // "user_id" → user, _, id
    let tokenized = tokenize("user_id");
    assert_eq!(tokenized.tokens.len(), 3);
    assert_eq!(tokenized.tokens[0].token_type, TokenType::Word);
    assert_eq!(tokenized.tokens[1].token_type, TokenType::Punctuation);
    assert_eq!(tokenized.token_text(&tokenized.tokens[1]), Some("_"));
    assert_eq!(tokenized.tokens[2].token_type, TokenType::Word);
}

#[test]
fn whitespace_only_yields_no_tokens() {
    assert!(tokenize("\n").is_empty());
    assert!(tokenize("   \t\r\n   ").is_empty());
}

#[test]
fn a_newline_b_sets_newline_flag() {
    let tokenized = tokenize("A\nB");
    assert_eq!(tokenized.tokens.len(), 2);
    assert!(!tokenized.tokens[0].first_token_after_newline);
    assert!(tokenized.tokens[1].first_token_after_newline);
}

#[test]
fn tokens_text_valid_interval_reconstructs_substring() {
    let tokenized = tokenize("Hello, world!");
    // Tokens: "Hello" , "," , "world" , "!"
    // Interval [0, 2) should be "Hello,"
    let s = tokens_text(&tokenized, TokenInterval::new(0, 2)).unwrap();
    assert_eq!(s, "Hello,");
    // Interval [2, 4) should be "world!"
    let s = tokens_text(&tokenized, TokenInterval::new(2, 4)).unwrap();
    assert_eq!(s, "world!");
    // Empty interval is the empty string.
    assert_eq!(
        tokens_text(&tokenized, TokenInterval::new(1, 1)).unwrap(),
        ""
    );
}

#[test]
fn tokens_text_invalid_interval_errors() {
    let tokenized = tokenize("one two three");
    let err = tokens_text(&tokenized, TokenInterval::new(0, 10)).unwrap_err();
    match err {
        TokenizerError::InvalidTokenInterval { start: 0, end: 10, total: 3 } => {}
        other => panic!("unexpected error: {other:?}"),
    }
    let err = tokens_text(&tokenized, TokenInterval::new(2, 1)).unwrap_err();
    assert!(matches!(
        err,
        TokenizerError::InvalidTokenInterval { .. }
    ));
}

#[test]
fn sentence_range_basic_period() {
    let text = "Hello world. Goodbye world.";
    let tokenized = tokenize(text);
    // Sentence 1 should be tokens [0, 3): "Hello world."
    let s1 = find_sentence_range(text, &tokenized.tokens, 0).unwrap();
    let span = tokens_text(&tokenized, s1).unwrap();
    assert_eq!(span, "Hello world.");
    // Sentence 2 should cover "Goodbye world."
    let s2 = find_sentence_range(text, &tokenized.tokens, s1.end_index).unwrap();
    let span = tokens_text(&tokenized, s2).unwrap();
    assert_eq!(span, "Goodbye world.");
}

#[test]
fn sentence_range_respects_abbreviations() {
    // "Dr. Smith arrived." — "Dr." must not end the sentence.
    let text = "Dr. Smith arrived.";
    let tokenized = tokenize(text);
    let s = find_sentence_range(text, &tokenized.tokens, 0).unwrap();
    assert_eq!(tokens_text(&tokenized, s).unwrap(), "Dr. Smith arrived.");
}

#[test]
fn sentence_range_absorbs_closing_quote() {
    // `He said "Hi."` — the closing `"` should be folded into the sentence.
    let text = "He said \"Hi.\" Then left.";
    let tokenized = tokenize(text);
    let s = find_sentence_range(text, &tokenized.tokens, 0).unwrap();
    let span = tokens_text(&tokenized, s).unwrap();
    assert_eq!(span, "He said \"Hi.\"");
}

#[test]
fn sentence_range_newline_capital_break() {
    // A newline followed by a capital letter on the next line should end
    // the sentence, even without a period.
    let text = "First line\nSecond line";
    let tokenized = tokenize(text);
    let s = find_sentence_range(text, &tokenized.tokens, 0).unwrap();
    let span = tokens_text(&tokenized, s).unwrap();
    // Python: ends at the *last* token of the first line (inclusive). Here
    // that's "line" at index 1, so interval is [0, 2).
    assert_eq!(span, "First line");
}

#[test]
fn sentence_range_empty_tokens_is_empty_interval() {
    let text = "";
    let tokenized = tokenize(text);
    let s = find_sentence_range(text, &tokenized.tokens, 0).unwrap();
    assert_eq!(s, TokenInterval::new(0, 0));
}

#[test]
fn sentence_range_out_of_bounds_start_errors() {
    let text = "hi";
    let tokenized = tokenize(text);
    let err = find_sentence_range(text, &tokenized.tokens, 99).unwrap_err();
    assert!(matches!(
        err,
        TokenizerError::SentenceStartOutOfRange { start: 99, total: 1 }
    ));
}

#[test]
fn regex_tokenizer_trait_roundtrip() {
    // Explicit trait dispatch path.
    let tok = RegexTokenizer::new();
    let tokenized = tok.tokenize("Hello!");
    assert_eq!(tokenized.tokens.len(), 2);
    assert_eq!(tokenized.tokens[0].token_type, TokenType::Word);
    assert_eq!(tokenized.tokens[1].token_type, TokenType::Punctuation);
}
