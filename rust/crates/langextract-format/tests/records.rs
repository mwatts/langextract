//! Integration tests mirroring `tests/resolver_test.py::ExtractOrderedEntitiesTest`.

use langextract_core::{AttributeValue, Extraction};
use langextract_format::{FormatError, FormatHandler, extract_ordered_extractions};
use pretty_assertions::assert_eq;
use serde_json::json;

/// Parse a JSON literal into the `Vec<ParsedRecord>` shape the function
/// accepts. Keeps test setup concise.
fn records(json_list: &serde_json::Value) -> Vec<serde_json::Map<String, serde_json::Value>> {
    json_list
        .as_array()
        .expect("records() input must be a JSON array")
        .iter()
        .map(|v| {
            v.as_object()
                .expect("each record must be a JSON object")
                .clone()
        })
        .collect()
}

const ATTR: &str = "_attributes";
const INDEX: Option<&str> = Some("_index");

fn extract_classes_texts_indices(out: &[Extraction]) -> Vec<(&str, &str, Option<i64>, Option<usize>)> {
    out.iter()
        .map(|e| {
            (
                e.extraction_class.as_str(),
                e.extraction_text.as_str(),
                e.extraction_index,
                e.group_index,
            )
        })
        .collect()
}

#[test]
fn valid_input_sorts_by_index() {
    // Mirrors the `valid_input` Python case.
    let input = records(&json!([
        {
            "medication": "Naprosyn",
            "medication_index": 4,
            "frequency": "as needed",
            "frequency_index": 5,
            "reason": "pain",
            "reason_index": 8,
        },
        {
            "medication": "prednisone",
            "medication_index": 5,
            "frequency": "daily",
            "frequency_index": 1,
        },
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![
            ("frequency", "daily", Some(1), Some(1)),
            ("medication", "Naprosyn", Some(4), Some(0)),
            ("frequency", "as needed", Some(5), Some(0)),
            ("medication", "prednisone", Some(5), Some(1)),
            ("reason", "pain", Some(8), Some(0)),
        ]
    );
}

#[test]
fn empty_input_returns_empty() {
    let out = extract_ordered_extractions(&[], ATTR, INDEX).unwrap();
    assert!(out.is_empty());
}

#[test]
fn mixed_index_order_sorts_stably() {
    let input = records(&json!([
        {
            "medication": "Ibuprofen",
            "medication_index": 2,
            "dosage": "400mg",
            "dosage_index": 1,
        },
        {
            "medication": "Acetaminophen",
            "medication_index": 1,
            "duration": "7 days",
            "duration_index": 2,
        },
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![
            ("dosage", "400mg", Some(1), Some(0)),
            ("medication", "Acetaminophen", Some(1), Some(1)),
            ("medication", "Ibuprofen", Some(2), Some(0)),
            ("duration", "7 days", Some(2), Some(1)),
        ]
    );
}

#[test]
fn missing_index_key_drops_the_extraction() {
    // "medication" has no "medication_index" → dropped; "dosage"
    // survives.
    let input = records(&json!([
        {
            "medication": "Aspirin",
            "dosage": "325mg",
            "dosage_index": 1,
        }
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![("dosage", "325mg", Some(1), Some(0))]
    );
}

#[test]
fn all_indices_missing_returns_empty_with_index_mode() {
    let input = records(&json!([
        {"medication": "Aspirin", "dosage": "325mg"},
        {"medication": "Ibuprofen", "dosage": "400mg"},
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert!(out.is_empty());
}

#[test]
fn duplicate_indices_preserve_source_order() {
    let input = records(&json!([
        {
            "medication": "Aspirin",
            "medication_index": 1,
            "dosage": "325mg",
            "dosage_index": 1,
            "form": "tablet",
            "form_index": 1,
        }
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    // All share extraction_index=1, so source order wins.
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![
            ("medication", "Aspirin", Some(1), Some(0)),
            ("dosage", "325mg", Some(1), Some(0)),
            ("form", "tablet", Some(1), Some(0)),
        ]
    );
}

#[test]
fn negative_indices_sort_ascending() {
    let input = records(&json!([
        {
            "medication": "Aspirin",
            "medication_index": -1,
            "dosage": "325mg",
            "dosage_index": -2,
        }
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![
            ("dosage", "325mg", Some(-2), Some(0)),
            ("medication", "Aspirin", Some(-1), Some(0)),
        ]
    );
}

#[test]
fn index_without_data_key_ignored() {
    // "medication_index" with no corresponding "medication" key is
    // just metadata — no extraction is emitted for it.
    let input = records(&json!([
        {
            "medication_index": 1,
            "dosage": "325mg",
            "dosage_index": 2,
        }
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![("dosage", "325mg", Some(2), Some(0))]
    );
}

#[test]
fn no_index_suffix_auto_increments() {
    let input = records(&json!([
        {"medication": "Aspirin"},
        {"medication": "Ibuprofen"},
        {"dosage": "325mg"},
        {"dosage": "400mg"},
    ]));
    let out = extract_ordered_extractions(&input, ATTR, None).unwrap();
    assert_eq!(
        extract_classes_texts_indices(&out),
        vec![
            ("medication", "Aspirin", Some(1), Some(0)),
            ("medication", "Ibuprofen", Some(2), Some(1)),
            ("dosage", "325mg", Some(3), Some(2)),
            ("dosage", "400mg", Some(4), Some(3)),
        ]
    );
}

#[test]
fn attributes_are_attached_to_extractions() {
    let input = records(&json!([
        {
            "patient": "Jane Doe",
            "patient_attributes": {
                "PERSON": "True",
                "IDENTIFIABLE": "True",
            },
        },
    ]));
    let out = extract_ordered_extractions(&input, ATTR, None).unwrap();
    assert_eq!(out.len(), 1);
    let attrs = out[0].attributes.as_ref().expect("attributes attached");
    assert_eq!(
        attrs.get("PERSON"),
        Some(&AttributeValue::Scalar("True".into()))
    );
    assert_eq!(
        attrs.get("IDENTIFIABLE"),
        Some(&AttributeValue::Scalar("True".into()))
    );
}

#[test]
fn non_integer_index_is_error() {
    let input = records(&json!([
        {
            "medication": "Aspirin",
            "medication_index": "not a number",
        }
    ]));
    let err = extract_ordered_extractions(&input, ATTR, INDEX).unwrap_err();
    assert!(matches!(err, FormatError::InvalidShape(_)));
}

#[test]
fn non_scalar_extraction_text_is_error() {
    let input = records(&json!([
        {
            "medication": {"nested": "object"},
            "medication_index": 1,
        }
    ]));
    let err = extract_ordered_extractions(&input, ATTR, INDEX).unwrap_err();
    assert!(matches!(err, FormatError::InvalidShape(_)));
}

#[test]
fn number_extraction_text_is_stringified() {
    let input = records(&json!([
        {
            "dosage": 325,
            "dosage_index": 1,
        }
    ]));
    let out = extract_ordered_extractions(&input, ATTR, INDEX).unwrap();
    assert_eq!(out.len(), 1);
    assert_eq!(out[0].extraction_text, "325");
}

#[test]
fn end_to_end_parse_then_extract() {
    // Drive the full FormatHandler → extract_ordered_extractions
    // path with realistic model output.
    let handler = FormatHandler::new();
    let model_output = r#"```json
{
  "extractions": [
    {
      "person": "Alice",
      "person_attributes": {"role": "engineer"},
      "person_index": 1
    },
    {
      "person": "Bob",
      "person_attributes": {"role": "manager"},
      "person_index": 2
    }
  ]
}
```"#;
    let records = handler.parse_output(model_output, None).unwrap();
    let out = extract_ordered_extractions(&records, ATTR, INDEX).unwrap();
    assert_eq!(out.len(), 2);
    assert_eq!(out[0].extraction_text, "Alice");
    assert_eq!(out[1].extraction_text, "Bob");
    assert_eq!(
        out[0]
            .attributes
            .as_ref()
            .and_then(|a| a.get("role")),
        Some(&AttributeValue::Scalar("engineer".into()))
    );
}
