//! The [`PromptTemplateStructured`] data type and its file loader.
//!
//! Port of `PromptTemplateStructured` and
//! `read_prompt_template_structured_from_file` from
//! `langextract/prompting.py`. The template is the user-authored part
//! of a prompt: a plain-language description of what to extract, plus
//! a list of few-shot [`ExampleData`] records. Runtime concerns
//! (fences, question prefixes, etc.) live on
//! [`QAPromptGenerator`](crate::QAPromptGenerator) instead.

use std::path::Path;

use langextract_core::{ExampleData, FormatType};
use serde::{Deserialize, Serialize};

use crate::error::PromptError;

/// A structured prompt template — the user-authored inputs that feed
/// into a [`QAPromptGenerator`](crate::QAPromptGenerator).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PromptTemplateStructured {
    /// Natural-language instructions describing the extraction task.
    /// Appears at the top of the rendered prompt.
    pub description: String,

    /// Few-shot examples illustrating the expected input → output
    /// behaviour. Each example is rendered into the prompt in order.
    #[serde(default)]
    pub examples: Vec<ExampleData>,
}

impl PromptTemplateStructured {
    /// Construct a template with just a description and no examples.
    #[must_use]
    pub fn new(description: impl Into<String>) -> Self {
        Self {
            description: description.into(),
            examples: Vec::new(),
        }
    }

    /// Add a single example. Returns `self` for chaining.
    #[must_use]
    pub fn with_example(mut self, example: ExampleData) -> Self {
        self.examples.push(example);
        self
    }
}

/// Load a [`PromptTemplateStructured`] from a file.
///
/// Port of `read_prompt_template_structured_from_file`. Supports
/// JSON and YAML depending on the `format` argument. The Python
/// version defaults to YAML; the Rust port makes the format
/// explicit because it is a deliberate choice, not a default.
///
/// # Errors
///
/// - [`PromptError::Read`] if the file can't be opened.
/// - [`PromptError::Parse`] if the file contents aren't valid YAML
///   or JSON for a `PromptTemplateStructured`.
pub fn read_prompt_template_from_file<P: AsRef<Path>>(
    path: P,
    format: FormatType,
) -> Result<PromptTemplateStructured, PromptError> {
    let path_ref = path.as_ref();
    let contents = std::fs::read_to_string(path_ref).map_err(|e| PromptError::Read {
        path: path_ref.display().to_string(),
        source: e,
    })?;
    match format {
        FormatType::Json => serde_json::from_str(&contents)
            .map_err(|e| PromptError::Parse(format!("JSON: {e}"))),
        FormatType::Yaml => serde_yml::from_str(&contents)
            .map_err(|e| PromptError::Parse(format!("YAML: {e}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use langextract_core::Extraction;
    use pretty_assertions::assert_eq;

    #[test]
    fn construct_with_chained_examples() {
        let t = PromptTemplateStructured::new("Extract stuff.")
            .with_example(ExampleData::new(
                "Alice is an engineer.",
                vec![Extraction::new("person", "Alice")],
            ))
            .with_example(ExampleData::new(
                "Bob builds bridges.",
                vec![Extraction::new("person", "Bob")],
            ));
        assert_eq!(t.description, "Extract stuff.");
        assert_eq!(t.examples.len(), 2);
    }

    #[test]
    fn read_from_yaml_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("template.yaml");
        std::fs::write(
            &path,
            "description: Extract entities.\nexamples:\n  - text: Alice.\n    extractions:\n      - extraction_class: person\n        extraction_text: Alice\n",
        )
        .unwrap();
        let t = read_prompt_template_from_file(&path, FormatType::Yaml).unwrap();
        assert_eq!(t.description, "Extract entities.");
        assert_eq!(t.examples.len(), 1);
        assert_eq!(t.examples[0].text, "Alice.");
        assert_eq!(t.examples[0].extractions[0].extraction_text, "Alice");
    }

    #[test]
    fn read_from_json_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("template.json");
        let json = r#"{
            "description": "Extract entities.",
            "examples": [
                {
                    "text": "Bob.",
                    "extractions": [
                        {"extraction_class": "person", "extraction_text": "Bob"}
                    ]
                }
            ]
        }"#;
        std::fs::write(&path, json).unwrap();
        let t = read_prompt_template_from_file(&path, FormatType::Json).unwrap();
        assert_eq!(t.description, "Extract entities.");
        assert_eq!(t.examples[0].extractions[0].extraction_text, "Bob");
    }

    #[test]
    fn missing_file_returns_read_error() {
        let err = read_prompt_template_from_file("/no/such/file.yaml", FormatType::Yaml)
            .unwrap_err();
        assert!(matches!(err, PromptError::Read { .. }));
    }
}
