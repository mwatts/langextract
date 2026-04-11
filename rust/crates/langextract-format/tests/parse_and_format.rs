//! Integration tests for [`FormatHandler`], mirroring
//! `tests/format_handler_test.py`.

use langextract_core::{AttributeMap, AttributeValue, Extraction, FormatType};
use langextract_format::{FormatError, FormatHandler};
use pretty_assertions::assert_eq;

// ---------- parameterized format_and_parse (condensed to three cases) ----------

#[test]
fn json_with_wrapper_and_fences_roundtrip() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(true)
        .build();

    let mut attrs = AttributeMap::new();
    attrs.insert("role".into(), AttributeValue::Scalar("engineer".into()));
    let extractions = vec![Extraction {
        attributes: Some(attrs),
        ..Extraction::new("person", "Alice")
    }];

    let formatted = handler.format_extraction_example(&extractions).unwrap();
    assert!(formatted.contains("```json"));
    assert!(formatted.contains("\"extractions\":"));
    assert!(formatted.contains("\"person\": \"Alice\""));

    // Parse a different model output into the same shape.
    let model_output = r#"Here is the result:
```json
{
  "extractions": [
    {"person": "Bob", "person_attributes": {"role": "manager"}}
  ]
}
```"#;
    let parsed = handler.parse_output(model_output, None).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].get("person").and_then(|v| v.as_str()), Some("Bob"));
}

#[test]
fn json_no_wrapper_no_fences() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(false)
        .use_fences(false)
        .build();

    let extractions = vec![Extraction::new("item", "book")];
    let formatted = handler.format_extraction_example(&extractions).unwrap();
    assert!(!formatted.contains("```"));
    assert!(formatted.contains("\"item\": \"book\""));

    let model_output = r#"[{"item": "pen", "item_attributes": {}}]"#;
    let parsed = handler.parse_output(model_output, None).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(parsed[0].get("item").and_then(|v| v.as_str()), Some("pen"));
}

#[test]
fn yaml_with_wrapper_and_fences() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Yaml)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(true)
        .build();

    let extractions = vec![Extraction::new("city", "Paris")];
    let formatted = handler.format_extraction_example(&extractions).unwrap();
    assert!(formatted.contains("```yaml"));
    assert!(formatted.contains("extractions:"));
    assert!(formatted.contains("city: Paris"));

    let model_output = "```yaml\nextractions:\n  - city: London\n    city_attributes: {}\n```";
    let parsed = handler.parse_output(model_output, None).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0].get("city").and_then(|v| v.as_str()),
        Some("London")
    );
}

// ---------- round-trip ----------

#[test]
fn json_with_wrapper_round_trips() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(true)
        .build();

    let mut attrs = AttributeMap::new();
    attrs.insert("key".into(), AttributeValue::Scalar("data".into()));
    let extractions = vec![Extraction {
        attributes: Some(attrs),
        ..Extraction::new("test", "value")
    }];

    let formatted = handler.format_extraction_example(&extractions).unwrap();
    let parsed = handler.parse_output(&formatted, None).unwrap();
    assert_eq!(parsed[0].get("test").and_then(|v| v.as_str()), Some("value"));
    assert_eq!(
        parsed[0]
            .get("test_attributes")
            .and_then(|v| v.get("key"))
            .and_then(|v| v.as_str()),
        Some("data"),
    );
}

#[test]
fn yaml_with_wrapper_round_trips() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Yaml)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(false)
        .build();

    let mut attrs = AttributeMap::new();
    attrs.insert("key".into(), AttributeValue::Scalar("data".into()));
    let extractions = vec![Extraction {
        attributes: Some(attrs),
        ..Extraction::new("test", "value")
    }];

    let formatted = handler.format_extraction_example(&extractions).unwrap();
    let parsed = handler.parse_output(&formatted, None).unwrap();
    assert_eq!(parsed[0].get("test").and_then(|v| v.as_str()), Some("value"));
}

// ---------- think-tag handling ----------

#[test]
fn think_tags_stripped_before_parsing() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(false)
        .build();

    let input_with_think =
        "<think>Let me analyze this text...</think>{\"extractions\": [{\"person\": \"Alice\"}]}";
    let parsed = handler.parse_output(input_with_think, None).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0].get("person").and_then(|v| v.as_str()),
        Some("Alice")
    );
}

#[test]
fn top_level_list_accepted_as_fallback() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(false)
        .build();

    let input_list = r#"[{"person": "Bob"}, {"person": "Carol"}]"#;
    let parsed = handler.parse_output(input_list, None).unwrap();
    assert_eq!(parsed.len(), 2);
    assert_eq!(
        parsed[0].get("person").and_then(|v| v.as_str()),
        Some("Bob")
    );
    assert_eq!(
        parsed[1].get("person").and_then(|v| v.as_str()),
        Some("Carol")
    );
}

#[test]
fn deepseek_r1_real_output() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(false)
        .build();

    let deepseek_output = "<think>\nAlright, so I need to extract people from the given text.\nI see John Smith is mentioned as an engineer.\n</think>\n{\"extractions\": [{\"person\": \"John Smith\"}]}";
    let parsed = handler.parse_output(deepseek_output, None).unwrap();
    assert_eq!(parsed.len(), 1);
    assert_eq!(
        parsed[0].get("person").and_then(|v| v.as_str()),
        Some("John Smith")
    );
}

// ---------- error cases ----------

#[test]
fn empty_input_errors() {
    let handler = FormatHandler::new();
    let err = handler.parse_output("", None).unwrap_err();
    assert!(matches!(err, FormatError::EmptyInput));
}

#[test]
fn strict_mode_rejects_top_level_list() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .use_wrapper(true)
        .wrapper_key("extractions")
        .use_fences(false)
        .build();

    let input = r#"[{"person": "Bob"}]"#;
    let err = handler.parse_output(input, Some(true)).unwrap_err();
    assert!(matches!(err, FormatError::InvalidShape(_)));
}

#[test]
fn strict_fences_rejects_multiple_candidates() {
    let handler = FormatHandler::builder()
        .format_type(FormatType::Json)
        .strict_fences(true)
        .build();

    let input = "```json\n{\"a\":1}\n```\nand:\n```json\n{\"b\":2}\n```";
    let err = handler.parse_output(input, None).unwrap_err();
    assert!(matches!(err, FormatError::MultipleFencedBlocks));
}
