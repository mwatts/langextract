//! End-to-end integration tests against a scripted fake
//! [`LanguageModel`]. Verifies that the full `extract()` pipeline
//! threads chunks through the model and lands grounded extractions
//! with absolute character offsets back into the source text.

use std::sync::Mutex;

use async_trait::async_trait;
use langextract::{
    AlignmentStatus, ExampleData, ExtractRequest, Extraction, InferError, InferenceParams,
    LanguageModel, ScoredOutput, extract,
};
use pretty_assertions::assert_eq;

/// A fake model that returns a canned response for each incoming
/// prompt, in the order the prompts are received. Tracks the prompts
/// it saw so tests can assert on them if they want.
#[derive(Debug)]
struct ScriptedModel {
    responses: Mutex<Vec<String>>,
    seen_prompts: Mutex<Vec<String>>,
}

impl ScriptedModel {
    const fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
            seen_prompts: Mutex::new(Vec::new()),
        }
    }

    fn take_seen_prompts(&self) -> Vec<String> {
        let mut guard = self.seen_prompts.lock().unwrap();
        std::mem::take(&mut *guard)
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
        for prompt in prompts {
            {
                let mut seen = self.seen_prompts.lock().unwrap();
                seen.push(prompt.clone());
            }
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

#[tokio::test]
async fn single_chunk_medications_fuzzy_alignment() {
    // One chunk that fits the default buffer. The model returns one
    // fenced JSON block with two medication extractions.
    let source = "Patient is prescribed Naprosyn and prednisone for treatment.";
    let model = ScriptedModel::new(vec![
        "```json\n\
         {\"extractions\": [\n\
           {\"medication\": \"Naprosyn\", \"medication_attributes\": {}},\n\
           {\"medication\": \"prednisone\", \"medication_attributes\": {}}\n\
         ]}\n\
         ```"
        .to_owned(),
    ]);

    let request = ExtractRequest {
        text: source.to_owned(),
        description: "Extract medications from clinical text.".to_owned(),
        examples: vec![ExampleData::new(
            "Took aspirin.",
            vec![Extraction::new("medication", "aspirin")],
        )],
        ..Default::default()
    };

    let result = extract(&model, request).await.unwrap();
    assert_eq!(result.text.as_deref(), Some(source));
    assert_eq!(result.extractions.len(), 2);

    let naprosyn = &result.extractions[0];
    assert_eq!(naprosyn.extraction_class, "medication");
    assert_eq!(naprosyn.extraction_text, "Naprosyn");
    assert_eq!(
        naprosyn.alignment_status,
        Some(AlignmentStatus::MatchExact)
    );
    let ci = naprosyn.char_interval.unwrap();
    // "Naprosyn" starts at byte 22 in the source string.
    assert_eq!(ci.start, 22);
    assert_eq!(ci.end, 30);
    assert_eq!(&source[ci.start..ci.end], "Naprosyn");

    let prednisone = &result.extractions[1];
    assert_eq!(prednisone.extraction_text, "prednisone");
    let ci = prednisone.char_interval.unwrap();
    assert_eq!(&source[ci.start..ci.end], "prednisone");
    assert_eq!(
        prednisone.alignment_status,
        Some(AlignmentStatus::MatchExact)
    );

    // The model saw exactly one prompt.
    let prompts = model.take_seen_prompts();
    assert_eq!(prompts.len(), 1);
    assert!(prompts[0].contains("Extract medications from clinical text."));
    assert!(prompts[0].contains("Patient is prescribed Naprosyn and prednisone for treatment."));
}

#[tokio::test]
async fn multi_chunk_offsets_are_absolute() {
    // Two chunks, one medication per chunk. Verify the second
    // extraction's char offsets are absolute into the ORIGINAL
    // source, not relative to its chunk. Both sentences fit
    // individually but not together.
    let source = "Alice gives Naprosyn. Bob gives prednisone.";
    let model = ScriptedModel::new(vec![
        "```json\n{\"extractions\": [{\"medication\": \"Naprosyn\", \"medication_attributes\": {}}]}\n```".to_owned(),
        "```json\n{\"extractions\": [{\"medication\": \"prednisone\", \"medication_attributes\": {}}]}\n```".to_owned(),
    ]);

    let request = ExtractRequest {
        text: source.to_owned(),
        description: "Extract medications.".to_owned(),
        examples: vec![],
        max_char_buffer: 25, // each sentence fits individually, both together don't
        ..Default::default()
    };

    let result = extract(&model, request).await.unwrap();
    assert_eq!(result.extractions.len(), 2);

    // Every extraction's char interval must slice back to its text
    // from the *original* source string.
    for ex in &result.extractions {
        let ci = ex.char_interval.expect("aligned");
        let sliced = &source[ci.start..ci.end];
        assert_eq!(
            sliced, ex.extraction_text,
            "char interval {:?} did not round-trip to extraction text {:?}",
            ci, ex.extraction_text
        );
    }

    // The second extraction must come after the first in the
    // original source coordinate space.
    let first_end = result.extractions[0].char_interval.unwrap().end;
    let second_start = result.extractions[1].char_interval.unwrap().start;
    assert!(
        second_start > first_end,
        "second extraction must come after first"
    );
    assert!(
        second_start > 20,
        "second extraction must be in the second sentence of the source"
    );
}

#[tokio::test]
async fn empty_model_response_is_not_fatal_for_subsequent_chunks() {
    // If one chunk's model output contains no extractions (empty
    // list), the pipeline should just skip that chunk and keep going.
    let source = "Nothing relevant here. Bob took Naprosyn today.";
    let model = ScriptedModel::new(vec![
        "```json\n{\"extractions\": []}\n```".to_owned(),
        "```json\n{\"extractions\": [{\"medication\": \"Naprosyn\", \"medication_attributes\": {}}]}\n```".to_owned(),
    ]);

    let request = ExtractRequest {
        text: source.to_owned(),
        description: "Extract medications.".to_owned(),
        examples: vec![],
        max_char_buffer: 25,
        ..Default::default()
    };

    let result = extract(&model, request).await.unwrap();
    assert_eq!(result.extractions.len(), 1);
    assert_eq!(result.extractions[0].extraction_text, "Naprosyn");
}
