//! Integration tests mirroring `tests/prompting_test.py`.

use langextract_core::{AttributeMap, AttributeValue, ExampleData, Extraction, FormatType};
use langextract_format::FormatHandler;
use langextract_prompting::{
    CONTEXT_PREFIX, ContextAwarePromptBuilder, PromptBuilder, PromptTemplateStructured,
    QAPromptGenerator, StatelessPromptBuilder,
};
use pretty_assertions::assert_eq;

fn attrs(pairs: &[(&str, &str)]) -> AttributeMap {
    let mut m = AttributeMap::new();
    for (k, v) in pairs {
        m.insert((*k).to_owned(), AttributeValue::Scalar((*v).to_owned()));
    }
    m
}

fn simple_generator() -> QAPromptGenerator {
    let template = PromptTemplateStructured::new("Extract entities.").with_example(
        ExampleData::new("Sample text.", vec![Extraction::new("entity", "Sample")]),
    );
    let handler = FormatHandler::builder()
        .format_type(FormatType::Yaml)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(true)
        .build();
    QAPromptGenerator::new(template, handler)
}

// ------------------------------------------------------------
// QAPromptGenerator::render
// ------------------------------------------------------------

#[test]
fn render_yaml_full_snapshot() {
    // Mirrors `test_generate_prompt`: a description + one example
    // with two medical_condition extractions, rendered with no
    // examples heading and no Q/A prefixes.
    let template = PromptTemplateStructured::new(
        "You are an assistant specialized in extracting key extractions from text.\n\
         Identify and extract important extractions such as people, places,\n\
         organizations, dates, and medical conditions mentioned in the text.\n\
         **Please ensure that the extractions are extracted in the same order as they\n\
         appear in the source text.**\n\
         Provide the extracted extractions in a structured YAML format.",
    )
    .with_example(ExampleData::new(
        "The patient was diagnosed with hypertension and diabetes.",
        vec![
            Extraction {
                attributes: Some(attrs(&[
                    ("chronicity", "chronic"),
                    ("system", "cardiovascular"),
                ])),
                ..Extraction::new("medical_condition", "hypertension")
            },
            Extraction {
                attributes: Some(attrs(&[
                    ("chronicity", "chronic"),
                    ("system", "endocrine"),
                ])),
                ..Extraction::new("medical_condition", "diabetes")
            },
        ],
    ));
    let handler = FormatHandler::builder()
        .format_type(FormatType::Yaml)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(true)
        .build();
    let generator = QAPromptGenerator::builder(template, handler)
        .examples_heading("")
        .question_prefix("")
        .answer_prefix("")
        .build();

    let actual = generator
        .render(
            "The patient reports chest pain and shortness of breath.",
            None,
        )
        .unwrap();

    // The exact YAML attribute-ordering depends on serde_yml, which
    // preserves the insertion order from our AttributeMap (a
    // BTreeMap → alphabetical). Instead of an exact string snapshot
    // we assert the structural pieces.
    assert!(actual.starts_with("You are an assistant specialized"));
    assert!(actual.contains("```yaml"));
    assert!(actual.contains("extractions:"));
    assert!(actual.contains("medical_condition: hypertension"));
    assert!(actual.contains("medical_condition: diabetes"));
    assert!(actual.contains("The patient was diagnosed with hypertension and diabetes."));
    assert!(actual.contains("The patient reports chest pain and shortness of breath."));
    // With empty prefixes, there should be no literal "Q:" or "A:"
    // markers.
    assert!(!actual.contains("Q:"));
    assert!(!actual.contains("A:"));
}

#[test]
fn render_places_additional_context_after_description() {
    let generator = simple_generator();
    let prompt = generator
        .render("input text", Some("Important context."))
        .unwrap();
    assert!(prompt.contains("Important context."));
    let desc_idx = prompt.find("Extract entities.").unwrap();
    let ctx_idx = prompt.find("Important context.").unwrap();
    assert!(ctx_idx > desc_idx);
}

#[test]
fn render_empty_additional_context_is_ignored() {
    let generator = simple_generator();
    let prompt = generator.render("input", Some("")).unwrap();
    assert!(!prompt.contains("\n\n\n"), "no double blank line gap");
    assert!(prompt.contains("input"));
}

#[test]
fn render_default_prefixes() {
    let generator = simple_generator();
    let prompt = generator.render("Bob is a manager", None).unwrap();
    assert!(prompt.contains("Examples"));
    assert!(prompt.contains("Q: Bob is a manager"));
    assert!(prompt.ends_with("A: "));
}

// ------------------------------------------------------------
// StatelessPromptBuilder
// ------------------------------------------------------------

#[test]
fn stateless_builder_renders_chunk_text() {
    let mut builder = StatelessPromptBuilder::new(simple_generator());
    let prompt = builder
        .build_prompt("Test input text.", "doc1", None)
        .unwrap();
    assert!(prompt.contains("Test input text."));
    assert!(prompt.contains("Extract entities."));
}

#[test]
fn stateless_builder_includes_additional_context() {
    let mut builder = StatelessPromptBuilder::new(simple_generator());
    let prompt = builder
        .build_prompt("Test input.", "doc1", Some("Important context here."))
        .unwrap();
    assert!(prompt.contains("Important context here."));
}

// ------------------------------------------------------------
// ContextAwarePromptBuilder
// ------------------------------------------------------------

#[test]
fn context_aware_first_chunk_has_no_previous() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(50));
    let prompt = builder
        .build_prompt("First chunk text.", "doc1", None)
        .unwrap();
    assert!(!prompt.contains(CONTEXT_PREFIX));
    assert!(prompt.contains("First chunk text."));
}

#[test]
fn context_aware_second_chunk_includes_previous() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(20));
    let _ = builder
        .build_prompt("First chunk ending.", "doc1", None)
        .unwrap();
    let second = builder
        .build_prompt("Second chunk text.", "doc1", None)
        .unwrap();
    assert!(second.contains(CONTEXT_PREFIX));
    assert!(second.contains("chunk ending."));
}

#[test]
fn context_aware_disabled_when_none() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), None);
    let _ = builder.build_prompt("First chunk.", "doc1", None).unwrap();
    let second = builder
        .build_prompt("Second chunk.", "doc1", None)
        .unwrap();
    assert!(!second.contains(CONTEXT_PREFIX));
}

#[test]
fn context_aware_isolates_per_document() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(50));
    let _ = builder
        .build_prompt("Doc A chunk one.", "docA", None)
        .unwrap();
    let _ = builder
        .build_prompt("Doc B chunk one.", "docB", None)
        .unwrap();
    let a2 = builder
        .build_prompt("Doc A chunk two.", "docA", None)
        .unwrap();
    let b2 = builder
        .build_prompt("Doc B chunk two.", "docB", None)
        .unwrap();
    assert!(a2.contains("Doc A chunk one"));
    assert!(!a2.contains("Doc B"));
    assert!(b2.contains("Doc B chunk one"));
    assert!(!b2.contains("Doc A"));
}

#[test]
fn context_aware_combines_previous_with_additional() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(30));
    let _ = builder
        .build_prompt("Previous chunk text.", "doc1", None)
        .unwrap();
    let prompt = builder
        .build_prompt("Current chunk.", "doc1", Some("Extra info here."))
        .unwrap();
    assert!(prompt.contains(CONTEXT_PREFIX));
    assert!(prompt.contains("Previous chunk text."));
    assert!(prompt.contains("Extra info here."));
}

#[test]
fn context_aware_clear_resets_state() {
    let mut builder = ContextAwarePromptBuilder::new(simple_generator(), Some(50));
    let _ = builder.build_prompt("First.", "doc1", None).unwrap();
    builder.clear();
    let second = builder.build_prompt("Second.", "doc1", None).unwrap();
    assert!(!second.contains(CONTEXT_PREFIX));
}

#[test]
fn context_aware_window_property() {
    let b1 = ContextAwarePromptBuilder::new(simple_generator(), None);
    assert_eq!(b1.context_window_chars(), None);
    let b2 = ContextAwarePromptBuilder::new(simple_generator(), Some(100));
    assert_eq!(b2.context_window_chars(), Some(100));
}
