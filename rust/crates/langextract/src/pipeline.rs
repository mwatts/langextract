//! The [`extract`] entry point and its supporting helpers.

use langextract_aligner::{AlignmentOptions, align_extraction_groups_with};
use langextract_chunking::ChunkIterator;
use langextract_core::{
    AnnotatedDocument, Document, DocumentId, InferenceParams, LanguageModel,
};
use langextract_format::{FormatHandler, extract_ordered_extractions};
use langextract_prompting::{
    ContextAwarePromptBuilder, PromptBuilder, PromptTemplateStructured, QAPromptGenerator,
    StatelessPromptBuilder,
};
use langextract_tokenizer::RegexTokenizer;

use crate::error::ExtractError;
use crate::request::ExtractRequest;

/// Run a single end-to-end extraction against the given language
/// model.
///
/// This is the top-level facade. It composes the focused sub-crates
/// into the full pipeline:
///
/// 1. Build a prompt template and format handler from the request.
/// 2. Create a prompt builder — either stateless or
///    [`ContextAwarePromptBuilder`] if
///    `context_window_chars` is set.
/// 3. Chunk the source text at sentence boundaries to fit
///    `max_char_buffer`.
/// 4. For each chunk, build the prompt, call the model, parse the
///    output into records, walk records into extractions, and align
///    them against the chunk text while restoring absolute offsets
///    via the chunk's `token_interval.start_index` /
///    `char_interval.start`.
/// 5. Merge every chunk's grounded extractions into a single
///    [`AnnotatedDocument`].
///
/// # Errors
///
/// Returns any error from the underlying crates as an
/// [`ExtractError`] — the variants are flat, so a `?`-chain works.
pub async fn extract(
    model: &dyn LanguageModel,
    request: ExtractRequest,
) -> Result<AnnotatedDocument, ExtractError> {
    let ExtractRequest {
        text,
        description,
        examples,
        max_char_buffer,
        format_type,
        use_wrapper,
        wrapper_key,
        attribute_suffix,
        extraction_index_suffix,
        use_fences,
        enable_fuzzy_alignment,
        fuzzy_alignment_threshold,
        accept_match_lesser,
        context_window_chars,
        document_id,
        additional_context,
    } = request;

    // 1. Build the format handler and prompt generator.
    let handler = FormatHandler::builder()
        .format_type(format_type)
        .use_wrapper(use_wrapper)
        .wrapper_key(wrapper_key)
        .use_fences(use_fences)
        .attribute_suffix(attribute_suffix.clone())
        .build();

    let template = PromptTemplateStructured {
        description,
        examples,
    };
    let generator = QAPromptGenerator::new(template, handler.clone());

    // 2. Pick a prompt builder based on whether cross-chunk context
    //    is requested.
    let mut prompt_builder: Box<dyn PromptBuilder> = if context_window_chars.is_some() {
        Box::new(ContextAwarePromptBuilder::new(
            generator,
            context_window_chars,
        ))
    } else {
        Box::new(StatelessPromptBuilder::new(generator))
    };

    // 3. Chunk the source. Tag the chunker with the document's id
    //    and additional context so the context-aware builder can
    //    isolate state per document.
    let effective_document_id = document_id
        .clone()
        .unwrap_or_else(DocumentId::new_random);
    let doc_id_str = effective_document_id.as_str().to_owned();
    let tokenizer = RegexTokenizer::new();

    // Build a Document so we can propagate both the id and the
    // additional_context onto every emitted chunk in one call.
    let source_doc = Document {
        text: text.clone(),
        document_id: Some(effective_document_id.clone()),
        additional_context: additional_context.clone(),
    };
    let chunk_iter = ChunkIterator::new(text.clone(), max_char_buffer)?
        .with_document(&source_doc);

    let mut all_grounded = Vec::new();
    let index_suffix_ref = extraction_index_suffix.as_deref();

    // 4. Per-chunk loop.
    for chunk in chunk_iter {
        // 4a. Build the prompt.
        let additional_from_chunk = chunk.additional_context.as_deref();
        let prompt = prompt_builder.build_prompt(
            &chunk.text,
            &doc_id_str,
            additional_from_chunk,
        )?;

        // 4b. Run the model.
        let outputs = model
            .infer(&[prompt], &InferenceParams::default())
            .await?;
        let Some(first_prompt_results) = outputs.into_iter().next() else {
            return Err(ExtractError::Inference(
                langextract_core::InferError::EmptyCompletions,
            ));
        };
        let Some(first_candidate) = first_prompt_results.into_iter().next() else {
            return Err(ExtractError::Inference(
                langextract_core::InferError::EmptyCompletions,
            ));
        };
        let raw_output = first_candidate.output;

        // 4c. Parse the output into records.
        let records = handler.parse_output(&raw_output, None)?;

        // 4d. Walk records into Extractions.
        let extractions =
            extract_ordered_extractions(&records, &attribute_suffix, index_suffix_ref)?;

        if extractions.is_empty() {
            continue;
        }

        // 4e. Align against the chunk's text, restoring absolute
        //     document offsets via the chunk's token/char start.
        let aligned = align_extraction_groups_with(
            vec![extractions],
            &chunk.text,
            &AlignmentOptions {
                token_offset: chunk.token_interval.start_index,
                char_offset: chunk.char_interval.start,
                enable_fuzzy_alignment,
                fuzzy_alignment_threshold,
                accept_match_lesser,
            },
            &tokenizer,
        )?;

        all_grounded.extend(aligned.into_iter().flatten());
    }

    // 5. Pack into an AnnotatedDocument.
    Ok(AnnotatedDocument {
        document_id: Some(effective_document_id),
        text: Some(text),
        extractions: all_grounded,
    })
}
