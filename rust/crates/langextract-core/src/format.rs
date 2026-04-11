//! Output format enumeration.
//!
//! Port of `FormatType` from `langextract/core/types.py`.

use serde::{Deserialize, Serialize};

/// Output format the LLM is expected to produce.
///
/// Extraction prompts always ask the model to return structured data in one
/// of these formats, typically wrapped in a Markdown code fence which the
/// resolver strips before parsing.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
#[non_exhaustive]
pub enum FormatType {
    /// JSON output. Default for most providers.
    #[default]
    Json,
    /// YAML output. Used when the provider or the user prefers it.
    Yaml,
}

impl FormatType {
    /// Markdown language tag used inside code fences for this format
    /// (`json` or `yaml`).
    #[must_use]
    pub const fn fence_tag(self) -> &'static str {
        match self {
            Self::Json => "json",
            Self::Yaml => "yaml",
        }
    }
}
