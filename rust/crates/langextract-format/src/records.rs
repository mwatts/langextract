//! Convert [`ParsedRecord`]s into [`Extraction`]s with Python-parity
//! ordering rules.
//!
//! Port of the `Resolver.extract_ordered_extractions` method from
//! `langextract/resolver.py`. This is the bridge between
//! [`FormatHandler::parse_output`](crate::FormatHandler::parse_output),
//! which gives you a sequence of flat key-value records, and the
//! higher-level pipeline that expects `Vec<Extraction>`.
//!
//! # Shape of a record
//!
//! A single record (one entry in the `extractions` list from the LLM)
//! is a flat JSON object containing **three kinds of keys**:
//!
//! 1. **Extraction-text keys.** These are the real payload: the key is
//!    the extraction class (`"medication"`, `"person"`, …) and the
//!    value is the extraction text. Values may be strings, numbers,
//!    or booleans — anything scalar is stringified.
//!
//! 2. **Attribute keys.** Formed as `<class><attribute_suffix>` (e.g.
//!    `"medication_attributes"`). Values are flat maps from attribute
//!    name to attribute value. Matching this pattern means "attach
//!    these attributes to the extraction of class `<class>` in the
//!    same record".
//!
//! 3. **Index keys.** Optional, formed as `<class><index_suffix>`
//!    (e.g. `"medication_index"`). Their integer value is used to
//!    order the extractions globally. Extractions whose class has no
//!    index key are silently dropped when ordering is enabled — this
//!    matches Python's behaviour and lets models decide which fields
//!    are "index-worthy".
//!
//! # Two ordering modes
//!
//! - **Explicit index suffix.** Pass `Some("_index")` (or whatever
//!   suffix the prompt uses). Each extraction takes its
//!   `extraction_index` from the corresponding `<class>_index` field;
//!   extractions without a matching index are dropped. The final
//!   output is sorted by `extraction_index`, **stably**, so records
//!   that share an index stay in their source order.
//!
//! - **Auto-increment.** Pass `None`. A single counter increments for
//!   every emitted extraction starting from 1, and the output
//!   preserves source order.

use langextract_core::{AttributeMap, AttributeValue, Extraction};
use serde_json::Value;

use crate::error::FormatError;
use crate::handler::ParsedRecord;

/// Convert parsed records into a flat, ordered list of [`Extraction`]s.
///
/// Port of `extract_ordered_extractions`. See the [module docs](crate::records)
/// for the three-kinds-of-keys data model and the two ordering modes.
///
/// # Arguments
///
/// - `records` — records as returned by
///   [`FormatHandler::parse_output`](crate::FormatHandler::parse_output).
/// - `attribute_suffix` — suffix identifying attribute sub-objects
///   (typically `"_attributes"` — see
///   [`langextract_core::ATTRIBUTE_SUFFIX`]).
/// - `extraction_index_suffix` — suffix identifying index fields
///   (typically `Some("_index")`). Pass `None` to auto-increment
///   instead.
///
/// # Errors
///
/// - [`FormatError::InvalidShape`] if an index field is not an integer,
///   if an extraction value is not a scalar, or if an attribute field
///   is not an object or null.
pub fn extract_ordered_extractions(
    records: &[ParsedRecord],
    attribute_suffix: &str,
    extraction_index_suffix: Option<&str>,
) -> Result<Vec<Extraction>, FormatError> {
    if records.is_empty() {
        return Ok(Vec::new());
    }

    let mut processed: Vec<Extraction> = Vec::new();
    let mut auto_index: i64 = 0;

    for (group_index, record) in records.iter().enumerate() {
        for (key, value) in record {
            // 1. Index-suffix key: validate it's an integer and skip.
            if let Some(suffix) = extraction_index_suffix {
                if key.ends_with(suffix) && !key.eq(suffix) {
                    if !is_integer(value) {
                        return Err(FormatError::shape(format!(
                            "Index field {key:?} must be an integer."
                        )));
                    }
                    continue;
                }
            }

            // 2. Attribute-suffix key: validate it's an object or null
            //    and skip (we'll read it via lookup below).
            if !attribute_suffix.is_empty()
                && key.ends_with(attribute_suffix)
                && !key.eq(attribute_suffix)
            {
                if !matches!(value, Value::Object(_) | Value::Null) {
                    return Err(FormatError::shape(format!(
                        "Attribute field {key:?} must be an object or null."
                    )));
                }
                continue;
            }

            // 3. Extraction-text key: must be a scalar.
            let Some(extraction_text) = value_to_extraction_text(value) else {
                return Err(FormatError::shape(format!(
                    "Extraction text for {key:?} must be a string, \
                     integer, or float.",
                )));
            };

            // Determine the extraction_index for this item.
            let extraction_index: i64 = if let Some(suffix) = extraction_index_suffix {
                let index_key = format!("{key}{suffix}");
                let Some(idx_value) = record.get(&index_key) else {
                    // No index for this class → silently drop the
                    // extraction. Matches Python's "continue" branch.
                    continue;
                };
                match idx_value.as_i64() {
                    Some(n) => n,
                    None => {
                        return Err(FormatError::shape(format!(
                            "Index field {index_key:?} must be an integer."
                        )));
                    }
                }
            } else {
                auto_index += 1;
                auto_index
            };

            // Attributes lookup.
            let attributes = if attribute_suffix.is_empty() {
                None
            } else {
                let attrs_key = format!("{key}{attribute_suffix}");
                match record.get(&attrs_key) {
                    Some(Value::Object(map)) => Some(json_object_to_attribute_map(map)),
                    Some(Value::Null) | None => None,
                    Some(other) => {
                        return Err(FormatError::shape(format!(
                            "Attribute field {attrs_key:?} must be an \
                             object or null, got {}.",
                            type_name(other),
                        )));
                    }
                }
            };

            processed.push(Extraction {
                extraction_index: Some(extraction_index),
                group_index: Some(group_index),
                attributes,
                ..Extraction::new(key.clone(), extraction_text)
            });
        }
    }

    // Sort by extraction_index, stably, so records sharing an index
    // preserve their source order — this matches Python's
    // `list.sort(key=attrgetter('extraction_index'))` on a list
    // with a stable sort.
    processed.sort_by_key(|e| e.extraction_index.unwrap_or(0));
    Ok(processed)
}

// ---------- helpers ----------

fn is_integer(v: &Value) -> bool {
    match v {
        Value::Number(n) => n.is_i64() || n.is_u64(),
        _ => false,
    }
}

/// Convert an extraction-text value to its string representation.
/// Returns `None` if the value is not a scalar (object, array, null).
fn value_to_extraction_text(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
    }
}

/// Turn a JSON object (attribute sub-record) into an [`AttributeMap`].
/// Unknown value types become `None` entries and are dropped.
fn json_object_to_attribute_map(obj: &serde_json::Map<String, Value>) -> AttributeMap {
    let mut out = AttributeMap::new();
    for (k, v) in obj {
        if let Some(av) = value_to_attribute_value(v) {
            out.insert(k.clone(), av);
        }
    }
    out
}

fn value_to_attribute_value(v: &Value) -> Option<AttributeValue> {
    match v {
        Value::String(s) => Some(AttributeValue::Scalar(s.clone())),
        Value::Number(n) => Some(AttributeValue::Scalar(n.to_string())),
        Value::Bool(b) => Some(AttributeValue::Scalar(b.to_string())),
        Value::Array(xs) => {
            let items: Vec<String> = xs.iter().filter_map(value_to_scalar_string).collect();
            Some(AttributeValue::List(items))
        }
        Value::Null | Value::Object(_) => None,
    }
}

fn value_to_scalar_string(v: &Value) -> Option<String> {
    match v {
        Value::String(s) => Some(s.clone()),
        Value::Number(n) => Some(n.to_string()),
        Value::Bool(b) => Some(b.to_string()),
        _ => None,
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

