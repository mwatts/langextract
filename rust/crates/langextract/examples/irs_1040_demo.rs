//! End-to-end demo: run the langextract pipeline on a real excerpt
//! from the IRS 1040 Instructions (2025 tax year), demonstrating
//! form-reference, tax-credit, and publication-reference extraction
//! with precise source grounding.
//!
//! The "LLM" here is a [`ScriptedModel`] that returns a canned
//! JSON response — hand-authored by reading the source excerpt the
//! same way an actual LLM would. This isolates the pipeline's
//! parse/ground/align machinery from any external API so the demo is
//! fully reproducible from `cargo run --example irs_1040_demo`.
//!
//! To reproduce with a real LLM instead, replace `ScriptedModel`
//! with a `CliLanguageModel` wrapping your coding-agent CLI of
//! choice — everything else stays the same.

use std::sync::Mutex;

use async_trait::async_trait;
use langextract::{
    AlignmentStatus, ExampleData, ExtractRequest, Extraction, FormatType, InferError,
    InferenceParams, LanguageModel, ScoredOutput, extract,
};

// Excerpt from IRS 2025 1040 Instructions, pp. 8. Public domain
// (U.S. government work). Trimmed and newline-joined to a clean
// paragraph form so the chunker produces one chunk with a generous
// buffer.
const SOURCE: &str = "\
Filing Requirements. Do You Have To File? Use Chart A, B, or C to see if you \
must file a return. U.S. citizens who lived in or had income from a U.S. \
territory should see Pub. 570. Residents of Puerto Rico can use Tax Topic 901 \
to see if they must file. Even if you do not otherwise have to file a return, \
you should file one to get a refund of any federal income tax withheld. You \
should also file if you are eligible for any of the following credits: Earned \
income credit; Additional child tax credit; American opportunity credit; \
Premium tax credit; Refundable adoption credit. See Pub. 501 for details. \
Also, see Pub. 501 if you do not have to file but received a Form 1099-B or \
Form 1099-DA. Requirement to reconcile advance payments of the premium tax \
credit: if you, your spouse, or a dependent was enrolled in coverage through \
the Marketplace for 2025 and advance payments of the premium tax credit were \
made, you must file a 2025 return and attach Form 8962. You should have \
received Form 1095-A from the Marketplace with information about your \
coverage and any advance payments.";

// ------------------------------------------------------------
// Canned LLM response
// ------------------------------------------------------------
//
// This is the "LLM output" — what a real language model (Claude,
// Gemini, Llama via Ollama, a coding-agent CLI, …) would return if
// given the SOURCE above together with a prompt asking it to extract
// form references, tax credits, publications, tax topics, and tax
// years.
//
// Each record uses the default `_attributes` suffix. The pipeline is
// configured below to use `_index` for ordering so the final output
// comes out in the order the fields are meaningful to a human
// reader, not the order they happened to appear in the source.
const CANNED_LLM_RESPONSE: &str = r#"```json
{
  "extractions": [
    {"chart_reference": "Chart A", "chart_reference_index": 1, "chart_reference_attributes": {"purpose": "determine filing requirement"}},
    {"chart_reference": "Chart B", "chart_reference_index": 2, "chart_reference_attributes": {"purpose": "determine filing requirement"}},
    {"chart_reference": "Chart C", "chart_reference_index": 3, "chart_reference_attributes": {"purpose": "determine filing requirement"}},
    {"publication": "Pub. 570", "publication_index": 4, "publication_attributes": {"topic": "U.S. territory residents"}},
    {"tax_topic": "Tax Topic 901", "tax_topic_index": 5, "tax_topic_attributes": {"topic": "Puerto Rico residents"}},
    {"tax_credit": "Earned income credit", "tax_credit_index": 6, "tax_credit_attributes": {"refundable": "yes"}},
    {"tax_credit": "Additional child tax credit", "tax_credit_index": 7, "tax_credit_attributes": {"refundable": "yes"}},
    {"tax_credit": "American opportunity credit", "tax_credit_index": 8, "tax_credit_attributes": {"refundable": "partially"}},
    {"tax_credit": "Premium tax credit", "tax_credit_index": 9, "tax_credit_attributes": {"refundable": "yes"}},
    {"tax_credit": "Refundable adoption credit", "tax_credit_index": 10, "tax_credit_attributes": {"refundable": "yes"}},
    {"publication": "Pub. 501", "publication_index": 11, "publication_attributes": {"topic": "dependents, standard deduction, filing information"}},
    {"form_reference": "Form 1099-B", "form_reference_index": 12, "form_reference_attributes": {"purpose": "proceeds from broker transactions"}},
    {"form_reference": "Form 1099-DA", "form_reference_index": 13, "form_reference_attributes": {"purpose": "digital asset transactions"}},
    {"tax_year": "2025", "tax_year_index": 14, "tax_year_attributes": {}},
    {"form_reference": "Form 8962", "form_reference_index": 15, "form_reference_attributes": {"purpose": "premium tax credit reconciliation"}},
    {"form_reference": "Form 1095-A", "form_reference_index": 16, "form_reference_attributes": {"purpose": "health insurance marketplace statement"}}
  ]
}
```"#;

// ------------------------------------------------------------
// Scripted fake LanguageModel
// ------------------------------------------------------------

#[derive(Debug)]
struct ScriptedModel {
    responses: Mutex<Vec<String>>,
}

impl ScriptedModel {
    const fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
        }
    }
}

#[async_trait]
impl LanguageModel for ScriptedModel {
    async fn infer(
        &self,
        prompts: &[String],
        _params: &InferenceParams,
    ) -> Result<Vec<Vec<ScoredOutput>>, InferError> {
        let mut out = Vec::with_capacity(prompts.len());
        for _ in prompts {
            let response = {
                let mut responses = self.responses.lock().unwrap();
                if responses.is_empty() {
                    return Err(InferError::EmptyCompletions);
                }
                responses.remove(0)
            };
            out.push(vec![ScoredOutput::unscored(response)]);
        }
        Ok(out)
    }
}

// ------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    println!("=== langextract / IRS 1040 Instructions demo ===\n");
    println!("Source length: {} bytes", SOURCE.len());
    println!("Source excerpt (first 200 chars):\n  {:?}\n", &SOURCE[..200.min(SOURCE.len())]);

    // One-shot extraction: the source fits in a single chunk, so
    // we feed a single canned response.
    let model = ScriptedModel::new(vec![CANNED_LLM_RESPONSE.to_owned()]);

    let request = ExtractRequest {
        text: SOURCE.to_owned(),
        description: "Extract IRS form references, tax credits, publications, \
                     tax topics, charts, and tax years from the following \
                     instructions text. Preserve the order of appearance."
            .to_owned(),
        examples: vec![ExampleData::new(
            "See Pub. 17 for Form 1040 instructions.",
            vec![
                Extraction::new("publication", "Pub. 17"),
                Extraction::new("form_reference", "Form 1040"),
            ],
        )],
        format_type: FormatType::Json,
        max_char_buffer: 2000, // whole excerpt in one chunk
        extraction_index_suffix: Some("_index".to_owned()),
        ..Default::default()
    };

    let result = extract(&model, request).await?;

    println!("Extractions: {}", result.extractions.len());
    println!();

    // Group by class for a nicer summary.
    let mut by_class: std::collections::BTreeMap<&str, Vec<&Extraction>> =
        std::collections::BTreeMap::new();
    for ex in &result.extractions {
        by_class
            .entry(ex.extraction_class.as_str())
            .or_default()
            .push(ex);
    }

    println!("─── Grouped by class ───");
    for (class, exs) in &by_class {
        println!("\n[{}] ({} found)", class, exs.len());
        for ex in exs {
            let status = match ex.alignment_status {
                Some(AlignmentStatus::MatchExact) => "EXACT",
                Some(AlignmentStatus::MatchGreater) => "GREATER",
                Some(AlignmentStatus::MatchLesser) => "LESSER",
                Some(AlignmentStatus::MatchFuzzy) => "FUZZY",
                Some(_) => "OTHER",
                None => "NONE ",
            };
            let ci = ex.char_interval;
            let span_display = ci.map_or_else(
                || "   —        —".to_owned(),
                |c| format!("{:4}..{:4}", c.start, c.end),
            );
            println!("  {status}  {span_display}  {:?}", ex.extraction_text);

            // Verify the char interval slices back to the extraction
            // text — this is the whole point of the grounding layer.
            if let Some(c) = ci {
                let sliced = &SOURCE[c.start..c.end];
                if sliced != ex.extraction_text {
                    println!(
                        "         ⚠ source slice {sliced:?} differs from extraction text"
                    );
                }
            }
        }
    }

    println!("\n─── Aligned in source order ───");
    let mut in_source_order: Vec<&Extraction> = result
        .extractions
        .iter()
        .filter(|e| e.char_interval.is_some())
        .collect();
    in_source_order.sort_by_key(|e| e.char_interval.unwrap().start);
    for ex in &in_source_order {
        let ci = ex.char_interval.unwrap();
        println!(
            "  {:>4}..{:<4}  [{:<15}]  {}",
            ci.start, ci.end, ex.extraction_class, ex.extraction_text
        );
    }

    let unaligned: Vec<&Extraction> = result
        .extractions
        .iter()
        .filter(|e| e.char_interval.is_none())
        .collect();
    if !unaligned.is_empty() {
        println!(
            "\n─── Unaligned ({} extractions) ───",
            unaligned.len()
        );
        for ex in &unaligned {
            println!(
                "  [{}]  {}  (LLM hallucinated or span not in source)",
                ex.extraction_class, ex.extraction_text
            );
        }
    }

    println!(
        "\nTotal: {} extractions, {} grounded, {} unaligned",
        result.extractions.len(),
        result.extractions.len() - unaligned.len(),
        unaligned.len()
    );

    Ok(())
}
