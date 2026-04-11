//! The [`FormatHandler`] type — central parser and formatter for
//! JSON/YAML LLM output in the langextract pipeline.
//!
//! Port of `core/format_handler.py::FormatHandler`. Legacy
//! compatibility helpers (`from_resolver_params`, `from_kwargs`) are
//! intentionally not ported — the Rust port starts from a clean slate.

use serde_json::{Map, Value};

use langextract_core::{
    ATTRIBUTE_SUFFIX, AttributeValue, EXTRACTIONS_KEY, Extraction, FormatType,
};

use crate::error::FormatError;
use crate::fence::{FencedBlock, find_fenced_blocks, language_tag_matches, strip_think_tags};

/// A parsed extraction record, as produced by [`FormatHandler::parse_output`].
///
/// This is a single flat JSON object which may contain one or more
/// `<class>` / `<class>_attributes` pairs plus optional index fields.
///
/// We use [`serde_json::Map`] rather than a newtype because the
/// downstream resolver (once ported) walks the keys by suffix and the
/// standard JSON value representation keeps the Rust ↔ Python JSON
/// semantics predictable.
pub type ParsedRecord = Map<String, Value>;

/// Output of the parser: a sequence of extraction records, in order.
pub type ParsedOutput = Vec<ParsedRecord>;

/// Handler for JSON/YAML prompt formatting and model-output parsing.
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag toggles an orthogonal Python-compatible behaviour; \
              packing them into a state-machine enum would muddy the 1:1 port \
              of core/format_handler.py without changing semantics"
)]
#[derive(Debug, Clone)]
pub struct FormatHandler {
    /// JSON or YAML.
    format_type: FormatType,

    /// Whether extractions are wrapped in a container object
    /// (`{"extractions": [...]}`) or a bare top-level list.
    use_wrapper: bool,

    /// The wrapper key. `Some(_)` when `use_wrapper` is `true`.
    wrapper_key: Option<String>,

    /// Whether to wrap formatted output in ``` ``` fences.
    use_fences: bool,

    /// Suffix used for attribute sub-objects on formatted extractions.
    attribute_suffix: String,

    /// Require exactly one fenced block in the input and reject
    /// everything else.
    strict_fences: bool,

    /// Whether to accept a top-level JSON/YAML list as a fallback
    /// when the wrapper key is missing.
    allow_top_level_list: bool,
}

impl Default for FormatHandler {
    fn default() -> Self {
        Self::new()
    }
}

impl FormatHandler {
    /// Construct a handler with Python-parity defaults: JSON output,
    /// wrapper on, `"extractions"` wrapper key, fences on,
    /// `_attributes` suffix, lenient fence matching, top-level list
    /// fallback allowed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            format_type: FormatType::Json,
            use_wrapper: true,
            wrapper_key: Some(EXTRACTIONS_KEY.to_owned()),
            use_fences: true,
            attribute_suffix: ATTRIBUTE_SUFFIX.to_owned(),
            strict_fences: false,
            allow_top_level_list: true,
        }
    }

    /// Start a builder. Prefer this over mutating individual fields.
    #[must_use]
    pub fn builder() -> FormatHandlerBuilder {
        FormatHandlerBuilder::default()
    }

    /// The output format type.
    #[must_use]
    pub const fn format_type(&self) -> FormatType {
        self.format_type
    }

    /// The configured wrapper key, if any.
    #[must_use]
    pub fn wrapper_key(&self) -> Option<&str> {
        self.wrapper_key.as_deref()
    }

    /// The configured attribute suffix.
    #[must_use]
    pub fn attribute_suffix(&self) -> &str {
        &self.attribute_suffix
    }

    /// Parse LLM output into a sequence of extraction records.
    ///
    /// Mirrors `FormatHandler.parse_output` in Python. The `strict`
    /// parameter overrides the handler's own lenient behaviour for
    /// this call — `Some(true)` enforces wrapper presence and rejects
    /// top-level lists, `Some(false)` forces lenient, `None` uses
    /// the handler's default (which is lenient).
    pub fn parse_output(
        &self,
        text: &str,
        strict: Option<bool>,
    ) -> Result<ParsedOutput, FormatError> {
        if text.is_empty() {
            return Err(FormatError::EmptyInput);
        }

        let content = self.extract_content(text)?;
        let parsed = self.parse_with_fallback(&content, strict.unwrap_or(false))?;

        // Reshape into a list of records, handling wrapper logic.
        self.reshape(parsed, strict.unwrap_or(false))
    }

    /// Format a list of [`Extraction`]s into a prompt example string.
    ///
    /// Mirrors `FormatHandler.format_extraction_example`.
    pub fn format_extraction_example(
        &self,
        extractions: &[Extraction],
    ) -> Result<String, FormatError> {
        // Build the list of one-record items.
        let items: Vec<Value> = extractions
            .iter()
            .map(|ex| {
                let mut m = Map::new();
                m.insert(
                    ex.extraction_class.clone(),
                    Value::String(ex.extraction_text.clone()),
                );
                let attrs_key = format!("{}{}", ex.extraction_class, self.attribute_suffix);
                let attrs_value = attributes_to_json(ex.attributes.as_ref());
                m.insert(attrs_key, attrs_value);
                Value::Object(m)
            })
            .collect();

        let payload = if let (true, Some(key)) = (self.use_wrapper, self.wrapper_key.as_deref()) {
            let mut obj = Map::new();
            obj.insert(key.to_owned(), Value::Array(items));
            Value::Object(obj)
        } else {
            Value::Array(items)
        };

        let serialized = match self.format_type {
            FormatType::Json => serde_json::to_string_pretty(&payload).map_err(|e| {
                FormatError::Parse {
                    format: "json",
                    source: Box::new(e),
                }
            })?,
            FormatType::Yaml => serde_yml::to_string(&payload).map_err(|e| FormatError::Parse {
                format: "yaml",
                source: Box::new(e),
            })?,
        };

        Ok(if self.use_fences {
            self.wrap_in_fence(serialized.trim())
        } else {
            serialized.trim().to_owned()
        })
    }

    // ---------- private helpers ----------

    fn wrap_in_fence(&self, content: &str) -> String {
        format!("```{}\n{}\n```", self.format_type.fence_tag(), content)
    }

    /// Port of `_extract_content`: pull the parseable payload out of a
    /// raw model response, handling fences.
    fn extract_content<'a>(&self, text: &'a str) -> Result<String, FormatError> {
        if !self.use_fences {
            return Ok(text.trim().to_owned());
        }

        let blocks = find_fenced_blocks(text);
        let candidates: Vec<&FencedBlock<'a>> = blocks
            .iter()
            .filter(|b| language_tag_matches(b.language, self.format_type))
            .collect();

        if self.strict_fences {
            return match candidates.len() {
                0 => Err(FormatError::FenceNotFound {
                    format: self.format_type.fence_tag(),
                }),
                1 => Ok(candidates[0].body.to_owned()),
                _ => Err(FormatError::MultipleFencedBlocks),
            };
        }

        match candidates.len() {
            0 => {
                // No language-matching candidate. If there is exactly
                // one unmatched fenced block, use it; otherwise fall
                // back to the raw text.
                if blocks.len() == 1 {
                    Ok(blocks[0].body.to_owned())
                } else if blocks.is_empty() {
                    Ok(text.trim().to_owned())
                } else {
                    Err(FormatError::FenceNotFound {
                        format: self.format_type.fence_tag(),
                    })
                }
            }
            1 => Ok(candidates[0].body.to_owned()),
            _ => Err(FormatError::MultipleFencedBlocks),
        }
    }

    /// Port of `_parse_with_fallback`: parse as JSON or YAML, with a
    /// secondary pass that strips `<think>...</think>` tags if the
    /// first attempt fails and we're not in strict mode.
    fn parse_with_fallback(&self, content: &str, strict: bool) -> Result<Value, FormatError> {
        match self.parse_value(content) {
            Ok(v) => Ok(v),
            Err(e) => {
                if strict {
                    return Err(e);
                }
                // Only retry if a <think> tag is actually present.
                let lower_has_think = contains_ignore_case(content, "<think>");
                if !lower_has_think {
                    return Err(e);
                }
                let stripped = strip_think_tags(content);
                let stripped_trim = stripped.trim();
                self.parse_value(stripped_trim)
            }
        }
    }

    fn parse_value(&self, content: &str) -> Result<Value, FormatError> {
        match self.format_type {
            FormatType::Json => serde_json::from_str(content).map_err(|e| FormatError::Parse {
                format: "json",
                source: Box::new(e),
            }),
            FormatType::Yaml => serde_yml::from_str(content).map_err(|e| FormatError::Parse {
                format: "yaml",
                source: Box::new(e),
            }),
        }
    }

    /// Port of the big reshape section at the bottom of `parse_output`.
    fn reshape(&self, parsed: Value, strict: bool) -> Result<ParsedOutput, FormatError> {
        let require_wrapper =
            self.wrapper_key.is_some() && (self.use_wrapper || strict);

        let items_value = match parsed {
            Value::Null => {
                return Err(if self.use_wrapper {
                    FormatError::shape(format!(
                        "Content must be a mapping with a {:?} key.",
                        self.wrapper_key.as_deref().unwrap_or("")
                    ))
                } else {
                    FormatError::shape("Content must be a list of extractions or a dict.")
                });
            }
            Value::Object(mut map) => {
                if require_wrapper {
                    let Some(key) = self.wrapper_key.as_deref() else {
                        return Err(FormatError::shape(
                            "wrapper required but wrapper_key is None",
                        ));
                    };
                    match map.remove(key) {
                        Some(v) => v,
                        None => {
                            return Err(FormatError::shape(format!(
                                "Content must contain a {key:?} key."
                            )));
                        }
                    }
                } else if let Some(v) = map.remove(EXTRACTIONS_KEY) {
                    v
                } else if let Some(key) = self.wrapper_key.as_deref() {
                    if let Some(v) = map.remove(key) {
                        v
                    } else {
                        // The whole object is treated as a single record.
                        return Ok(vec![map]);
                    }
                } else {
                    return Ok(vec![map]);
                }
            }
            Value::Array(list) => {
                if require_wrapper && (strict || !self.allow_top_level_list) {
                    return Err(FormatError::shape(format!(
                        "Content must be a mapping with a {:?} key.",
                        self.wrapper_key.as_deref().unwrap_or("")
                    )));
                }
                if strict && self.use_wrapper {
                    return Err(FormatError::shape("Strict mode requires a wrapper object."));
                }
                if !self.allow_top_level_list {
                    return Err(FormatError::shape("Top-level list is not allowed."));
                }
                Value::Array(list)
            }
            other => {
                return Err(FormatError::shape(format!(
                    "Expected list or dict, got {}",
                    type_name(&other)
                )));
            }
        };

        let Value::Array(items) = items_value else {
            return Err(FormatError::shape(
                "The extractions must be a sequence (list) of mappings.",
            ));
        };

        let mut out = Vec::with_capacity(items.len());
        for item in items {
            match item {
                Value::Object(map) => {
                    // All keys are already `String` in `serde_json::Map`,
                    // so the Python "all keys must be strings" check
                    // is automatically satisfied.
                    out.push(map);
                }
                _ => {
                    return Err(FormatError::shape(
                        "Each item in the sequence must be a mapping.",
                    ));
                }
            }
        }
        Ok(out)
    }
}

/// Builder for [`FormatHandler`].
#[expect(
    clippy::struct_excessive_bools,
    reason = "mirrors FormatHandler field-for-field"
)]
#[derive(Debug, Clone)]
pub struct FormatHandlerBuilder {
    format_type: FormatType,
    use_wrapper: bool,
    wrapper_key: Option<String>,
    use_fences: bool,
    attribute_suffix: String,
    strict_fences: bool,
    allow_top_level_list: bool,
}

impl Default for FormatHandlerBuilder {
    fn default() -> Self {
        Self {
            format_type: FormatType::Json,
            use_wrapper: true,
            wrapper_key: Some(EXTRACTIONS_KEY.to_owned()),
            use_fences: true,
            attribute_suffix: ATTRIBUTE_SUFFIX.to_owned(),
            strict_fences: false,
            allow_top_level_list: true,
        }
    }
}

impl FormatHandlerBuilder {
    /// Set the output format type.
    #[must_use]
    pub const fn format_type(mut self, fmt: FormatType) -> Self {
        self.format_type = fmt;
        self
    }

    /// Enable or disable wrapping extractions in a container object.
    #[must_use]
    pub fn use_wrapper(mut self, use_wrapper: bool) -> Self {
        self.use_wrapper = use_wrapper;
        if !use_wrapper {
            self.wrapper_key = None;
        } else if self.wrapper_key.is_none() {
            self.wrapper_key = Some(EXTRACTIONS_KEY.to_owned());
        }
        self
    }

    /// Set the wrapper key name (`"extractions"` by default).
    #[must_use]
    pub fn wrapper_key(mut self, key: impl Into<String>) -> Self {
        self.wrapper_key = Some(key.into());
        self.use_wrapper = true;
        self
    }

    /// Enable or disable Markdown fences around formatted output.
    #[must_use]
    pub const fn use_fences(mut self, use_fences: bool) -> Self {
        self.use_fences = use_fences;
        self
    }

    /// Override the attribute-key suffix (`"_attributes"` by default).
    #[must_use]
    pub fn attribute_suffix(mut self, suffix: impl Into<String>) -> Self {
        self.attribute_suffix = suffix.into();
        self
    }

    /// Require exactly one valid fenced block in the input.
    #[must_use]
    pub const fn strict_fences(mut self, strict: bool) -> Self {
        self.strict_fences = strict;
        self
    }

    /// Allow a top-level list as a fallback when the wrapper is missing.
    #[must_use]
    pub const fn allow_top_level_list(mut self, allow: bool) -> Self {
        self.allow_top_level_list = allow;
        self
    }

    /// Finalize the builder.
    #[must_use]
    pub fn build(self) -> FormatHandler {
        FormatHandler {
            format_type: self.format_type,
            use_wrapper: self.use_wrapper,
            wrapper_key: self.wrapper_key,
            use_fences: self.use_fences,
            attribute_suffix: self.attribute_suffix,
            strict_fences: self.strict_fences,
            allow_top_level_list: self.allow_top_level_list,
        }
    }
}

// ---------- free helpers ----------

fn attributes_to_json(attrs: Option<&langextract_core::AttributeMap>) -> Value {
    let Some(attrs) = attrs else {
        return Value::Object(Map::new());
    };
    let mut m = Map::with_capacity(attrs.len());
    for (k, v) in attrs {
        m.insert(k.clone(), attribute_value_to_json(v));
    }
    Value::Object(m)
}

fn attribute_value_to_json(v: &AttributeValue) -> Value {
    match v {
        AttributeValue::Scalar(s) => Value::String(s.clone()),
        AttributeValue::List(xs) => Value::Array(xs.iter().cloned().map(Value::String).collect()),
    }
}

const fn type_name(v: &Value) -> &'static str {
    match v {
        Value::Null => "null",
        Value::Bool(_) => "bool",
        Value::Number(_) => "number",
        Value::String(_) => "string",
        Value::Array(_) => "array",
        Value::Object(_) => "object",
    }
}

fn contains_ignore_case(haystack: &str, needle: &str) -> bool {
    if needle.is_empty() {
        return true;
    }
    let h = haystack.as_bytes();
    let n = needle.as_bytes();
    h.windows(n.len()).any(|w| w.eq_ignore_ascii_case(n))
}
