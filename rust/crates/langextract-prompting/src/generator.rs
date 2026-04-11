//! [`QAPromptGenerator`] — renders a [`PromptTemplateStructured`] into
//! the actual text a language model sees.
//!
//! Port of `QAPromptGenerator` from `langextract/prompting.py`. The
//! Python class is a dataclass with a `render` method; the Rust port
//! exposes it as a struct with a builder, because several knobs
//! (examples heading, question/answer prefixes) are optional and the
//! builder keeps construction readable.

use langextract_core::ExampleData;
use langextract_format::FormatHandler;

use crate::error::PromptError;
use crate::template::PromptTemplateStructured;

/// Default heading above the few-shot examples block.
pub const DEFAULT_EXAMPLES_HEADING: &str = "Examples";

/// Default prefix for the question line (the input text).
pub const DEFAULT_QUESTION_PREFIX: &str = "Q: ";

/// Default prefix for the answer line (the model's expected output).
pub const DEFAULT_ANSWER_PREFIX: &str = "A: ";

/// Generates question-answer prompts from a structured template.
#[derive(Debug, Clone)]
pub struct QAPromptGenerator {
    template: PromptTemplateStructured,
    format_handler: FormatHandler,
    examples_heading: String,
    question_prefix: String,
    answer_prefix: String,
}

impl QAPromptGenerator {
    /// Construct a generator with Python-parity defaults.
    #[must_use]
    pub fn new(template: PromptTemplateStructured, format_handler: FormatHandler) -> Self {
        Self {
            template,
            format_handler,
            examples_heading: DEFAULT_EXAMPLES_HEADING.to_owned(),
            question_prefix: DEFAULT_QUESTION_PREFIX.to_owned(),
            answer_prefix: DEFAULT_ANSWER_PREFIX.to_owned(),
        }
    }

    /// Start a builder for overriding the prefixes and heading.
    #[must_use]
    pub fn builder(
        template: PromptTemplateStructured,
        format_handler: FormatHandler,
    ) -> QAPromptGeneratorBuilder {
        QAPromptGeneratorBuilder {
            template,
            format_handler,
            examples_heading: DEFAULT_EXAMPLES_HEADING.to_owned(),
            question_prefix: DEFAULT_QUESTION_PREFIX.to_owned(),
            answer_prefix: DEFAULT_ANSWER_PREFIX.to_owned(),
        }
    }

    /// Borrow the template.
    #[must_use]
    pub const fn template(&self) -> &PromptTemplateStructured {
        &self.template
    }

    /// Borrow the format handler.
    #[must_use]
    pub const fn format_handler(&self) -> &FormatHandler {
        &self.format_handler
    }

    /// Format a single [`ExampleData`] into its question/answer text
    /// form. The question is the example's raw input text; the answer
    /// is the example extractions serialized by the
    /// [`FormatHandler`].
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::Format`] if the handler fails to
    /// serialize the extractions (should not normally happen for
    /// in-memory data).
    pub fn format_example_as_text(&self, example: &ExampleData) -> Result<String, PromptError> {
        let answer = self
            .format_handler
            .format_extraction_example(&example.extractions)?;
        Ok(format!(
            "{q_prefix}{question}\n{a_prefix}{answer}\n",
            q_prefix = self.question_prefix,
            question = example.text,
            a_prefix = self.answer_prefix,
        ))
    }

    /// Render the full prompt for a given `question` (typically the
    /// current chunk text) and optional additional context.
    ///
    /// The layout mirrors `prompting.py::QAPromptGenerator.render`:
    ///
    /// ```text
    /// <description>
    ///
    /// [<additional_context>]      // only if non-empty
    ///
    /// <examples_heading>          // only if template has examples
    /// <Q: example 1 text>
    /// <A: example 1 answer>
    ///
    /// <Q: example 2 text>
    /// <A: example 2 answer>
    ///
    /// <Q: question>
    /// <A: >                       // empty — the model fills this in
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`PromptError::Format`] if any example fails to
    /// serialize.
    pub fn render(
        &self,
        question: &str,
        additional_context: Option<&str>,
    ) -> Result<String, PromptError> {
        let mut lines: Vec<String> = Vec::new();
        lines.push(format!("{}\n", self.template.description));

        if let Some(ctx) = additional_context {
            if !ctx.is_empty() {
                lines.push(format!("{ctx}\n"));
            }
        }

        if !self.template.examples.is_empty() {
            lines.push(self.examples_heading.clone());
            for ex in &self.template.examples {
                lines.push(self.format_example_as_text(ex)?);
            }
        }

        lines.push(format!("{}{question}", self.question_prefix));
        lines.push(self.answer_prefix.clone());
        Ok(lines.join("\n"))
    }
}

/// Builder for [`QAPromptGenerator`].
#[derive(Debug, Clone)]
pub struct QAPromptGeneratorBuilder {
    template: PromptTemplateStructured,
    format_handler: FormatHandler,
    examples_heading: String,
    question_prefix: String,
    answer_prefix: String,
}

impl QAPromptGeneratorBuilder {
    /// Override the heading that appears above the examples block.
    /// Pass an empty string to suppress the heading entirely.
    #[must_use]
    pub fn examples_heading(mut self, heading: impl Into<String>) -> Self {
        self.examples_heading = heading.into();
        self
    }

    /// Override the prefix for question lines (default `"Q: "`).
    #[must_use]
    pub fn question_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.question_prefix = prefix.into();
        self
    }

    /// Override the prefix for answer lines (default `"A: "`).
    #[must_use]
    pub fn answer_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.answer_prefix = prefix.into();
        self
    }

    /// Finalize the builder.
    #[must_use]
    pub fn build(self) -> QAPromptGenerator {
        QAPromptGenerator {
            template: self.template,
            format_handler: self.format_handler,
            examples_heading: self.examples_heading,
            question_prefix: self.question_prefix,
            answer_prefix: self.answer_prefix,
        }
    }
}
