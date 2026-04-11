//! Core data types used throughout the extraction pipeline.
//!
//! Port of `langextract/core/data.py`. Preserves the Python module's public
//! shape but replaces Python idioms with idiomatic Rust:
//!
//! | Python | Rust |
//! |---|---|
//! | `@dataclass` | `#[derive(...)]` struct |
//! | `enum.Enum` | `enum` |
//! | `dict[str, str \| list[str]]` | [`AttributeMap`] (typed enum value) |
//! | Lazy auto-generated `document_id` | [`DocumentId::new_random`] on first access |
//!
//! Lazy tokenization (the Python `_tokenized_text` field) is **not** part of
//! this type; tokenization lives in a separate crate and is held by the
//! pipeline, not on the data object. That matches how a Rust consumer would
//! reason about ownership.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Default key used to wrap the extractions list inside a container
/// object, e.g. `{"extractions": [...]}`. Ported from `EXTRACTIONS_KEY`
/// in `langextract/core/data.py`.
pub const EXTRACTIONS_KEY: &str = "extractions";

/// Default suffix for attribute sub-objects on a formatted extraction.
///
/// For extraction class `"person"`, the attribute key becomes
/// `"person_attributes"`. Ported from `ATTRIBUTE_SUFFIX` in
/// `langextract/core/data.py`.
pub const ATTRIBUTE_SUFFIX: &str = "_attributes";

/// A half-open character interval `[start, end)` into some source text.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CharInterval {
    /// Starting character index (inclusive).
    pub start: usize,
    /// Ending character index (exclusive).
    pub end: usize,
}

impl CharInterval {
    /// Construct a new interval.
    ///
    /// # Panics
    ///
    /// Panics if `start > end`.
    #[must_use]
    pub fn new(start: usize, end: usize) -> Self {
        assert!(start <= end, "CharInterval start must be <= end");
        Self { start, end }
    }

    /// Length of the interval in characters.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end - self.start
    }

    /// Returns `true` if this interval contains no characters.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start == self.end
    }
}

/// A half-open interval `[start_index, end_index)` over a token sequence.
///
/// Lives in `langextract-core` (not the tokenizer crate) because
/// [`Extraction`] carries one, and keeping it here avoids a core →
/// tokenizer dependency cycle. The tokenizer crate re-exports it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub struct TokenInterval {
    /// Index of the first token in the interval.
    pub start_index: usize,
    /// Index one past the last token in the interval.
    pub end_index: usize,
}

impl TokenInterval {
    /// Construct a new interval.
    #[must_use]
    pub const fn new(start_index: usize, end_index: usize) -> Self {
        Self {
            start_index,
            end_index,
        }
    }

    /// Number of tokens covered.
    #[must_use]
    pub const fn len(self) -> usize {
        self.end_index - self.start_index
    }

    /// Whether the interval covers no tokens.
    #[must_use]
    pub const fn is_empty(self) -> bool {
        self.start_index == self.end_index
    }
}

/// How well an extraction's text aligned against the source document.
///
/// Port of `AlignmentStatus` in `core/data.py`. The resolver attaches this
/// to each [`Extraction`] after alignment.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum AlignmentStatus {
    /// Exact character-level match.
    MatchExact,
    /// A superset of the predicted span matched.
    MatchGreater,
    /// A subset of the predicted span matched.
    MatchLesser,
    /// Fuzzy (non-exact) match above the configured threshold.
    MatchFuzzy,
}

/// A value in an extraction's attribute map.
///
/// Python's `dict[str, str | list[str]]` becomes a typed two-variant enum
/// so callers don't need to pattern-match on `serde_json::Value`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    /// A single scalar string value.
    Scalar(String),
    /// A list of string values.
    List(Vec<String>),
}

impl From<&str> for AttributeValue {
    fn from(s: &str) -> Self {
        Self::Scalar(s.to_owned())
    }
}

impl From<String> for AttributeValue {
    fn from(s: String) -> Self {
        Self::Scalar(s)
    }
}

impl From<Vec<String>> for AttributeValue {
    fn from(v: Vec<String>) -> Self {
        Self::List(v)
    }
}

/// Type alias for an extraction's attribute map.
pub type AttributeMap = BTreeMap<String, AttributeValue>;

/// A single extraction emitted by the LLM and resolved against the source.
///
/// Port of the `Extraction` dataclass in `core/data.py`. All positional
/// metadata (`char_interval`, `alignment_status`, indices) is optional and
/// is filled in by the resolver after the raw LLM output is parsed.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Extraction {
    /// The class label for this extraction (`"character"`, `"medication"`, ...).
    pub extraction_class: String,

    /// Verbatim text of the extraction as produced by the LLM.
    pub extraction_text: String,

    /// Character span in the source document that this extraction aligns to.
    /// `None` if the resolver could not locate the text in the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub char_interval: Option<CharInterval>,

    /// Token span in the source document that this extraction aligns to.
    /// Populated by the aligner alongside [`char_interval`](Self::char_interval).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_interval: Option<TokenInterval>,

    /// How well the extraction aligned to the source.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alignment_status: Option<AlignmentStatus>,

    /// Index of this extraction in the output list.
    ///
    /// Signed because Python's `extraction_index` has no lower bound
    /// and tests exercise negative indices for "lower priority" or
    /// "placeholder" records. Resolvers that assign their own
    /// positive, monotonically increasing indices can ignore the
    /// sign.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub extraction_index: Option<i64>,

    /// Index of the group this extraction belongs to (used for co-referring
    /// extractions from the same output record).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub group_index: Option<usize>,

    /// Free-form human description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,

    /// Typed attribute map attached to this extraction.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub attributes: Option<AttributeMap>,
}

impl Extraction {
    /// Minimal constructor. Use struct literal syntax for the other fields.
    #[must_use]
    pub fn new(class: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            extraction_class: class.into(),
            extraction_text: text.into(),
            char_interval: None,
            token_interval: None,
            alignment_status: None,
            extraction_index: None,
            group_index: None,
            description: None,
            attributes: None,
        }
    }
}

/// Strongly typed wrapper for a document identifier (`M-NEWTYPE`).
///
/// Documents carry an ID for tracking through the pipeline. The Python
/// implementation auto-generates `doc_{uuid4-hex[:8]}` on first access; we
/// expose both an explicit constructor and an auto-generator to avoid the
/// hidden-mutation pattern.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(String);

impl DocumentId {
    /// Wrap an existing identifier.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Generate a new random identifier of the form `doc_xxxxxxxx`.
    #[must_use]
    pub fn new_random() -> Self {
        let hex = Uuid::new_v4().simple().to_string();
        // Match Python: `doc_{uuid4.hex[:8]}`.
        Self(format!("doc_{}", &hex[..8]))
    }

    /// Borrow the underlying string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// Consume the newtype and return the inner `String`.
    #[must_use]
    pub fn into_inner(self) -> String {
        self.0
    }
}

impl std::fmt::Display for DocumentId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// An input document to be annotated.
///
/// Port of the `Document` dataclass. The Python implementation lazily
/// computes `tokenized_text` on first access; in Rust we keep the data type
/// pure and let the pipeline hold tokenized state separately.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Document {
    /// Raw document text.
    pub text: String,

    /// Optional document identifier. If `None`, the pipeline will mint one
    /// via [`DocumentId::new_random`] before processing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_id: Option<DocumentId>,

    /// Additional context to supplement the prompt instructions for this
    /// specific document (e.g., a note to the model about the document's
    /// origin or structure).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub additional_context: Option<String>,
}

impl Document {
    /// Construct a new document from raw text.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            document_id: None,
            additional_context: None,
        }
    }

    /// Return the document's ID, generating one if necessary.
    #[must_use]
    pub fn ensure_id(&mut self) -> DocumentId {
        self.document_id
            .get_or_insert_with(DocumentId::new_random)
            .clone()
    }
}

/// A document with extractions attached.
///
/// Port of `AnnotatedDocument`. This is what [`crate::model::LanguageModel`]
/// ultimately contributes to; the resolver populates the `extractions` list
/// after the LLM output has been parsed and aligned.
#[derive(Debug, Clone, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct AnnotatedDocument {
    /// Identifier of the source document.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub document_id: Option<DocumentId>,

    /// Source text, if retained (the Python type allows `None` here too).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub text: Option<String>,

    /// Extractions found in this document.
    #[serde(default)]
    pub extractions: Vec<Extraction>,
}

/// A single few-shot example for structured prompting.
///
/// Port of `ExampleData`. Consumers construct these in their own code and
/// pass them to the prompt builder.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExampleData {
    /// Example input text.
    pub text: String,

    /// Expected extractions for the example.
    #[serde(default)]
    pub extractions: Vec<Extraction>,
}

impl ExampleData {
    /// Construct an example from text and a set of extractions.
    #[must_use]
    pub fn new(text: impl Into<String>, extractions: Vec<Extraction>) -> Self {
        Self {
            text: text.into(),
            extractions,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn char_interval_len_and_empty() {
        let ci = CharInterval::new(3, 10);
        assert_eq!(ci.len(), 7);
        assert!(!ci.is_empty());
        assert!(CharInterval::new(5, 5).is_empty());
    }

    #[test]
    #[should_panic(expected = "start must be <= end")]
    fn char_interval_panics_on_inverted() {
        let _ = CharInterval::new(10, 3);
    }

    #[test]
    fn document_id_generates_prefixed_random() {
        let id = DocumentId::new_random();
        let s = id.as_str();
        assert!(s.starts_with("doc_"), "expected doc_ prefix, got {s}");
        assert_eq!(s.len(), 4 + 8);
    }

    #[test]
    fn document_ensure_id_is_stable() {
        let mut doc = Document::new("hello world");
        let id1 = doc.ensure_id();
        let id2 = doc.ensure_id();
        assert_eq!(id1, id2);
    }

    #[test]
    fn extraction_round_trip_json() {
        let mut attrs = AttributeMap::new();
        attrs.insert("role".into(), AttributeValue::Scalar("hero".into()));
        attrs.insert(
            "traits".into(),
            AttributeValue::List(vec!["brave".into(), "curious".into()]),
        );
        let e = Extraction {
            attributes: Some(attrs),
            ..Extraction::new("character", "Juliet")
        };
        let j = serde_json::to_string(&e).unwrap();
        let back: Extraction = serde_json::from_str(&j).unwrap();
        assert_eq!(e, back);
    }
}
