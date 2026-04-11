//! Fence and think-tag scanning.
//!
//! Port of the private regex-driven helpers in
//! `core/format_handler.py`. Implemented manually to avoid a regex
//! crate dependency; the patterns are simple enough (two triple-
//! backtick markers with an optional language tag on the opening one).
//!
//! The scanner finds **all** fenced code blocks in an input string and
//! returns them as [`FencedBlock`]s so callers can filter by language
//! tag (e.g. keep only `json`/`yaml`/`yml` blocks) and apply strict or
//! lenient rules on top.

/// A single fenced code block found in an input string.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FencedBlock<'a> {
    /// The language tag that followed the opening fence, if any. For
    /// `` ```json\n... ``` `` this is `Some("json")`. For `` ```\n... ``` ``
    /// it is `None`.
    pub language: Option<&'a str>,
    /// The body of the fenced block, with leading/trailing whitespace
    /// trimmed.
    pub body: &'a str,
    /// Byte offset of the start of the opening fence in the input.
    pub start: usize,
    /// Byte offset one past the end of the closing fence in the input.
    pub end: usize,
}

/// Find every fenced code block in `text`. The blocks are returned in
/// the order they appear.
///
/// A fenced block is:
///
/// 1. Three backticks, optionally followed by a language tag made of
///    `[A-Za-z0-9_+-]+`.
/// 2. An optional trailing newline (consumed).
/// 3. A body, up to:
/// 4. The next occurrence of three backticks.
///
/// Nested fences are not supported — the first closing fence wins. If
/// an opening fence has no matching closing fence it is ignored.
#[must_use]
pub fn find_fenced_blocks(text: &str) -> Vec<FencedBlock<'_>> {
    let bytes = text.as_bytes();
    let mut blocks = Vec::new();
    let mut pos: usize = 0;

    while let Some(rel) = find_from(bytes, pos, b"```") {
        let open = rel;
        // Parse the optional language tag.
        let lang_start = open + 3;
        let mut lang_end = lang_start;
        while lang_end < bytes.len() {
            let b = bytes[lang_end];
            if b.is_ascii_alphanumeric() || b == b'_' || b == b'+' || b == b'-' {
                lang_end += 1;
            } else {
                break;
            }
        }
        let language = if lang_end > lang_start {
            Some(&text[lang_start..lang_end])
        } else {
            None
        };

        // Skip optional whitespace + a single newline between tag and body.
        let mut body_start = lang_end;
        while body_start < bytes.len() {
            let b = bytes[body_start];
            if b == b' ' || b == b'\t' {
                body_start += 1;
            } else {
                break;
            }
        }
        if body_start < bytes.len() && bytes[body_start] == b'\r' {
            body_start += 1;
        }
        if body_start < bytes.len() && bytes[body_start] == b'\n' {
            body_start += 1;
        }

        // Find the next ``` after the body start.
        let Some(close) = find_from(bytes, body_start, b"```") else {
            // Orphan opening fence — advance past it and continue.
            pos = open + 3;
            continue;
        };

        // Trim the body of leading/trailing whitespace.
        let body_raw = &text[body_start..close];
        let body = body_raw.trim();

        blocks.push(FencedBlock {
            language,
            body,
            start: open,
            end: close + 3,
        });
        pos = close + 3;
    }

    blocks
}

/// Return `true` if the language tag is acceptable for the given
/// [`FormatType`](langextract_core::FormatType). An absent tag is
/// always accepted (matching Python's lenient mode).
#[must_use]
pub fn language_tag_matches(language: Option<&str>, fmt: langextract_core::FormatType) -> bool {
    use langextract_core::FormatType;
    let Some(lang) = language else {
        return true;
    };
    let tag = lang.trim().to_ascii_lowercase();
    match fmt {
        FormatType::Json => tag == "json",
        FormatType::Yaml => tag == "yaml" || tag == "yml",
    }
}

/// Strip any `<think>...</think>` blocks (case-insensitive) from the input.
///
/// Reasoning models such as DeepSeek-R1 and `QwQ` emit these as chain-of-
/// thought traces before the actual JSON/YAML output. The strip is
/// **only** a fallback — the caller should first try parsing the raw
/// content, and only apply this if parsing fails.
#[must_use]
pub fn strip_think_tags(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let bytes = input.as_bytes();
    let open = b"<think>";
    let close = b"</think>";
    let mut pos = 0;
    while pos < bytes.len() {
        if let Some(rel) = find_ignore_case_from(bytes, pos, open) {
            // Copy everything before the tag.
            out.push_str(&input[pos..rel]);
            // Find matching close tag.
            let after_open = rel + open.len();
            if let Some(close_pos) = find_ignore_case_from(bytes, after_open, close) {
                // Skip the closing tag + any trailing whitespace.
                let mut next = close_pos + close.len();
                while next < bytes.len() {
                    let b = bytes[next];
                    if b == b' ' || b == b'\t' || b == b'\n' || b == b'\r' {
                        next += 1;
                    } else {
                        break;
                    }
                }
                pos = next;
            } else {
                // Unclosed — keep the rest verbatim and stop.
                out.push_str(&input[rel..]);
                return out;
            }
        } else {
            out.push_str(&input[pos..]);
            return out;
        }
    }
    out
}

// ---------- primitive byte search helpers ----------

fn find_from(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from >= haystack.len() || needle.is_empty() {
        return None;
    }
    // Naive but correct — input sizes are tiny (LLM outputs).
    haystack[from..]
        .windows(needle.len())
        .position(|w| w == needle)
        .map(|idx| from + idx)
}

fn find_ignore_case_from(haystack: &[u8], from: usize, needle: &[u8]) -> Option<usize> {
    if from >= haystack.len() || needle.is_empty() {
        return None;
    }
    haystack[from..]
        .windows(needle.len())
        .position(|w| w.eq_ignore_ascii_case(needle))
        .map(|idx| from + idx)
}

#[cfg(test)]
mod tests {
    use super::*;
    use langextract_core::FormatType;
    use pretty_assertions::assert_eq;

    #[test]
    fn finds_single_json_block() {
        let input = "Here:\n```json\n{\"a\": 1}\n```\n";
        let blocks = find_fenced_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].language, Some("json"));
        assert_eq!(blocks[0].body, "{\"a\": 1}");
    }

    #[test]
    fn finds_block_without_language_tag() {
        let input = "```\n{\"a\": 1}\n```";
        let blocks = find_fenced_blocks(input);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].language, None);
        assert_eq!(blocks[0].body, "{\"a\": 1}");
    }

    #[test]
    fn finds_multiple_blocks_in_order() {
        let input = "first:\n```json\n{\"x\":1}\n```\nsecond:\n```yaml\ny: 2\n```";
        let blocks = find_fenced_blocks(input);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].language, Some("json"));
        assert_eq!(blocks[1].language, Some("yaml"));
    }

    #[test]
    fn orphan_opening_fence_is_ignored() {
        let blocks = find_fenced_blocks("some ``` text");
        assert!(blocks.is_empty());
    }

    #[test]
    fn language_tag_matches_cases() {
        assert!(language_tag_matches(Some("json"), FormatType::Json));
        assert!(language_tag_matches(Some("JSON"), FormatType::Json));
        assert!(!language_tag_matches(Some("yaml"), FormatType::Json));
        assert!(language_tag_matches(Some("yml"), FormatType::Yaml));
        // Absent tag is always acceptable.
        assert!(language_tag_matches(None, FormatType::Json));
    }

    #[test]
    fn strip_think_tags_basic() {
        let input = "<think>reasoning...</think>\n{\"a\": 1}";
        assert_eq!(strip_think_tags(input), "{\"a\": 1}");
    }

    #[test]
    fn strip_think_tags_case_insensitive() {
        let input = "<THINK>r</THINK>{\"a\": 1}";
        assert_eq!(strip_think_tags(input), "{\"a\": 1}");
    }

    #[test]
    fn strip_think_tags_leaves_input_unchanged_if_no_tags() {
        let input = "{\"a\": 1}";
        assert_eq!(strip_think_tags(input), input);
    }

    #[test]
    fn strip_think_tags_handles_multiple() {
        let input = "<think>a</think>{\"x\": 1}<think>b</think>";
        assert_eq!(strip_think_tags(input), "{\"x\": 1}");
    }
}
