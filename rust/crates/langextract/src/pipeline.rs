//! The [`extract`] entry point and its concurrent per-chunk runner.
//!
//! This module composes every focused sub-crate into one async
//! function. See `crate::batch::extract_batch` for the
//! document-level parallel variant.

use std::sync::Arc;
use std::time::Instant;

use futures::StreamExt;
use futures::stream;
use langextract_aligner::{
    AlignmentOptions, align_extraction_groups_with_diagnostics_and,
};
use langextract_chunking::{ChunkIterator, TextChunk};
use langextract_core::{
    AnnotatedDocument, Document, DocumentId, Extraction, InferenceParams, LanguageModel,
};
use langextract_format::{FormatHandler, extract_ordered_extractions};
use langextract_prompting::{
    ContextAwarePromptBuilder, PromptBuilder, PromptTemplateStructured, QAPromptGenerator,
    StatelessPromptBuilder,
};
use langextract_tokenizer::RegexTokenizer;
use tracing::{Instrument, debug, info, info_span, warn};

use crate::cache::{CacheKey, ChunkCache};
use crate::error::ExtractError;
use crate::report::{ChunkReport, DocumentReport, unaligned_reason_label};
use crate::request::ExtractRequest;
use crate::retry::{RetryPolicy, is_transient_format, is_transient_infer};

/// Run a single end-to-end extraction against the given language
/// model and return the grounded [`AnnotatedDocument`].
///
/// Use [`extract_with_report`] if you also want the observability
/// report (recommended for batch runs and production pipelines).
///
/// # Errors
///
/// Returns any underlying sub-crate error flattened through
/// [`ExtractError`].
pub async fn extract(
    model: &dyn LanguageModel,
    request: ExtractRequest,
) -> Result<AnnotatedDocument, ExtractError> {
    let (doc, _report) = extract_with_report(model, request).await?;
    Ok(doc)
}

/// Run a single end-to-end extraction and return both the grounded
/// document and its [`DocumentReport`].
///
/// # Errors
///
/// See [`extract`].
#[expect(
    clippy::too_many_lines,
    reason = "verbatim composition of the prompt, chunk, infer, parse, align stages"
)]
pub async fn extract_with_report(
    model: &dyn LanguageModel,
    request: ExtractRequest,
) -> Result<(AnnotatedDocument, DocumentReport), ExtractError> {
    let start = Instant::now();

    // Destructure the request once so the compiler checks we use
    // every field.
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
        chunk_concurrency,
        retry_policy,
        chunk_cache,
        fuzzy_safeguards,
    } = request;

    let effective_document_id = document_id
        .clone()
        .unwrap_or_else(DocumentId::new_random);
    let doc_id_str = effective_document_id.as_str().to_owned();

    let span = info_span!(
        "langextract.extract",
        document_id = %doc_id_str,
        source_bytes = text.len(),
    );
    let _enter = span.enter();
    info!(source_bytes = text.len(), "starting extraction");

    // 1. Build the format handler up front (shared across all chunks).
    let handler = FormatHandler::builder()
        .format_type(format_type)
        .use_wrapper(use_wrapper)
        .wrapper_key(wrapper_key.clone())
        .use_fences(use_fences)
        .attribute_suffix(attribute_suffix.clone())
        .build();

    // Fingerprint used for cache keys — any of these changing
    // invalidates cached responses.
    let schema_fingerprint = format!(
        "{}|{}|{}|{}|{}",
        format_type_tag(format_type),
        use_wrapper,
        wrapper_key,
        use_fences,
        attribute_suffix,
    );

    // 2. Prompt builder: stateful context-aware OR stateless. The
    //    context-aware variant requires mutation across calls, so
    //    with chunk concurrency > 1 we warn and fall back to
    //    stateless — we can't interleave state safely.
    let mut prompt_builder_box: Box<dyn PromptBuilder> =
        if context_window_chars.is_some() && chunk_concurrency == 1 {
            let template = PromptTemplateStructured {
                description: description.clone(),
                examples: examples.clone(),
            };
            let generator = QAPromptGenerator::new(template, handler.clone());
            Box::new(ContextAwarePromptBuilder::new(
                generator,
                context_window_chars,
            ))
        } else {
            if context_window_chars.is_some() && chunk_concurrency > 1 {
                warn!(
                    "context_window_chars set but chunk_concurrency > 1 — \
                     falling back to stateless prompt builder (cross-chunk \
                     context cannot be interleaved safely across parallel tasks)",
                );
            }
            let template = PromptTemplateStructured {
                description: description.clone(),
                examples: examples.clone(),
            };
            let generator = QAPromptGenerator::new(template, handler.clone());
            Box::new(StatelessPromptBuilder::new(generator))
        };

    // 3. Build a Document so chunks inherit id + additional_context.
    let source_doc = Document {
        text: text.clone(),
        document_id: Some(effective_document_id.clone()),
        additional_context: additional_context.clone(),
    };
    let chunks: Vec<TextChunk> = ChunkIterator::new(text.clone(), max_char_buffer)?
        .with_document(&source_doc)
        .collect();
    let chunk_total = chunks.len();
    debug!(chunks = chunk_total, "chunked document");

    // 4. Build a per-chunk prompt list up front. Context-aware
    //    builder must run serially so we do it here, before the
    //    concurrent LLM loop.
    let mut chunk_prompts: Vec<(TextChunk, String)> = Vec::with_capacity(chunk_total);
    for chunk in chunks {
        let prompt = prompt_builder_box.build_prompt(
            &chunk.text,
            &doc_id_str,
            chunk.additional_context.as_deref(),
        )?;
        chunk_prompts.push((chunk, prompt));
    }

    // 5. Concurrent per-chunk execution.
    //
    // Each chunk's work:
    //   a. Check cache for a hit on (description, schema, chunk text).
    //   b. On miss, call model.infer with retry/backoff.
    //   c. Parse output → records → extractions (with retry on format errors).
    //   d. Align against chunk text with absolute offsets.
    //
    // Results are collected into per-chunk ChunkExecution structs and
    // then reduced into a single AnnotatedDocument + DocumentReport.
    let shared = Arc::new(SharedState {
        handler,
        description: description.clone(),
        schema_fingerprint,
        attribute_suffix: attribute_suffix.clone(),
        extraction_index_suffix: extraction_index_suffix.clone(),
        enable_fuzzy_alignment,
        fuzzy_alignment_threshold,
        accept_match_lesser,
        fuzzy_safeguards,
        retry_policy,
        cache: chunk_cache,
    });

    let mut report = DocumentReport::new(doc_id_str.clone());
    let mut all_grounded: Vec<Extraction> = Vec::new();

    // futures::stream::iter + buffer_unordered handles the
    // concurrency. We use FuturesOrdered first for simplicity in
    // preserving chunk_index ordering in reports; for true
    // speedups on large docs, switch to buffer_unordered.
    let concurrency = chunk_concurrency.max(1);

    // Build one future per chunk. Each future is Send because the
    // model is a trait object and the shared state is Arc.
    let jobs = chunk_prompts
        .into_iter()
        .enumerate()
        .map(|(idx, (chunk, prompt))| {
            let shared = Arc::clone(&shared);
            let model_ref: &dyn LanguageModel = model;
            async move {
                let chunk_span = info_span!(
                    "langextract.chunk",
                    chunk_index = idx,
                    chunk_bytes = chunk.text.len(),
                );
                process_chunk(chunk, prompt, idx, &shared, model_ref)
                    .instrument(chunk_span)
                    .await
            }
        });

    let mut stream = stream::iter(jobs).buffered(concurrency);
    while let Some(result) = stream.next().await {
        let (chunk_report, mut grounded) = result?;
        report.total_emitted += chunk_report.emitted;
        report.total_grounded += chunk_report.grounded;
        report.total_retries += chunk_report.retries;
        if chunk_report.cache_hit {
            report.cache_hits += 1;
        } else {
            report.cache_misses += 1;
        }
        for (reason_label, count) in &chunk_report.unaligned_reasons {
            *report
                .unaligned_reason_histogram
                .entry(*reason_label)
                .or_insert(0) += count;
        }
        for ex in &grounded {
            report.alignment_counts.add(ex.alignment_status);
        }
        report.chunks.push(chunk_report);
        all_grounded.append(&mut grounded);
    }

    report.elapsed = start.elapsed();

    info!(
        chunks = report.chunks.len(),
        total_emitted = report.total_emitted,
        total_grounded = report.total_grounded,
        grounding_rate = report.grounding_rate(),
        elapsed_ms = u64::try_from(report.elapsed.as_millis()).unwrap_or(u64::MAX),
        cache_hits = report.cache_hits,
        cache_misses = report.cache_misses,
        total_retries = report.total_retries,
        "extraction complete",
    );

    let doc = AnnotatedDocument {
        document_id: Some(effective_document_id),
        text: Some(text),
        extractions: all_grounded,
    };
    Ok((doc, report))
}

/// State shared across every per-chunk task in a single document run.
/// Wrapped in an `Arc` by the caller; fields are read-only except
/// through `&Mutex` on the cache.
struct SharedState {
    handler: FormatHandler,
    description: String,
    schema_fingerprint: String,
    attribute_suffix: String,
    extraction_index_suffix: Option<String>,
    enable_fuzzy_alignment: bool,
    fuzzy_alignment_threshold: f32,
    accept_match_lesser: bool,
    fuzzy_safeguards: langextract_aligner::FuzzySafeguards,
    retry_policy: RetryPolicy,
    cache: Arc<dyn ChunkCache>,
}

impl std::fmt::Debug for SharedState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SharedState")
            .field("description_len", &self.description.len())
            .field("schema_fingerprint", &self.schema_fingerprint)
            .finish_non_exhaustive()
    }
}

/// Process a single chunk: cache → model → parse → records → align.
/// Returns the per-chunk report plus the chunk's grounded
/// extractions with absolute offsets.
async fn process_chunk(
    chunk: TextChunk,
    prompt: String,
    chunk_index: usize,
    shared: &SharedState,
    model: &dyn LanguageModel,
) -> Result<(ChunkReport, Vec<Extraction>), ExtractError> {
    let chunk_start = Instant::now();
    let mut retries: u32 = 0;
    let cache_key = CacheKey::from(
        &shared.description,
        &shared.schema_fingerprint,
        &chunk.text,
    );

    // 1. Cache lookup.
    let (raw_output, cache_hit) = if let Some(hit) = shared.cache.get(&cache_key) {
        debug!(cache_key = cache_key.as_str(), "cache hit");
        (hit, true)
    } else {
        let response =
            call_model_with_retries(model, &prompt, &shared.retry_policy, chunk_index, &mut retries)
                .await?;
        shared.cache.put(&cache_key, response.clone());
        (response, false)
    };

    // 2. Parse and walk records. Retry on format errors as long as
    //    we haven't already exhausted the retry budget.
    let extractions = match parse_and_walk(shared, &raw_output) {
        Ok(list) => list,
        Err(e) if is_transient_format_error(&e) && retries < shared.retry_policy.max_retries => {
            debug!(?e, "format error; retrying model call");
            retries += 1;
            let response = call_model_with_retries(
                model,
                &prompt,
                &shared.retry_policy,
                chunk_index,
                &mut retries,
            )
            .await?;
            shared.cache.put(&cache_key, response.clone());
            parse_and_walk(shared, &response)?
        }
        Err(e) => return Err(e),
    };

    let emitted = extractions.len();

    // 3. Align.
    let report_from_aligner = align_extraction_groups_with_diagnostics_and(
        vec![extractions],
        &chunk.text,
        &AlignmentOptions {
            token_offset: chunk.token_interval.start_index,
            char_offset: chunk.char_interval.start,
            enable_fuzzy_alignment: shared.enable_fuzzy_alignment,
            fuzzy_alignment_threshold: shared.fuzzy_alignment_threshold,
            accept_match_lesser: shared.accept_match_lesser,
            fuzzy_safeguards: shared.fuzzy_safeguards,
        },
        &RegexTokenizer::new(),
    )?;

    let grounded_count = report_from_aligner.grounded_count();
    let mut per_chunk_reason_hist: std::collections::HashMap<&'static str, usize> =
        std::collections::HashMap::new();
    for reason in report_from_aligner.unaligned_reasons.values() {
        *per_chunk_reason_hist
            .entry(unaligned_reason_label(reason))
            .or_insert(0) += 1;
    }

    let grounded: Vec<Extraction> = report_from_aligner
        .groups
        .into_iter()
        .flatten()
        .filter(|e| e.char_interval.is_some())
        .collect();

    let chunk_report = ChunkReport {
        chunk_index,
        chunk_bytes: chunk.text.len(),
        emitted,
        grounded: grounded_count,
        retries,
        cache_hit,
        elapsed: chunk_start.elapsed(),
        unaligned_reasons: per_chunk_reason_hist,
    };
    Ok((chunk_report, grounded))
}

const fn is_transient_format_error(err: &ExtractError) -> bool {
    match err {
        ExtractError::Format(fe) => is_transient_format(fe),
        _ => false,
    }
}

/// Parse a raw model response into Extractions.
fn parse_and_walk(
    shared: &SharedState,
    raw_output: &str,
) -> Result<Vec<Extraction>, ExtractError> {
    let records = shared.handler.parse_output(raw_output, None)?;
    let extractions = extract_ordered_extractions(
        &records,
        &shared.attribute_suffix,
        shared.extraction_index_suffix.as_deref(),
    )?;
    Ok(extractions)
}

/// Call `model.infer` with retries on transient failures.
async fn call_model_with_retries(
    model: &dyn LanguageModel,
    prompt: &str,
    policy: &RetryPolicy,
    chunk_index: usize,
    retries: &mut u32,
) -> Result<String, ExtractError> {
    let mut attempt: u32 = 0;
    loop {
        let params = InferenceParams::default();
        let result = model.infer(std::slice::from_ref(&prompt.to_owned()), &params).await;
        match result {
            Ok(outputs) => {
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
                return Ok(first_candidate.output);
            }
            Err(err) => {
                if !is_transient_infer(&err) || attempt >= policy.max_retries {
                    return Err(ExtractError::Inference(err));
                }
                attempt += 1;
                *retries += 1;
                let jitter_seed =
                    (chunk_index as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ u64::from(attempt);
                let backoff = policy.backoff_for(attempt, jitter_seed);
                debug!(
                    attempt,
                    backoff_ms = u64::try_from(backoff.as_millis()).unwrap_or(u64::MAX),
                    ?err,
                    "retrying transient inference error"
                );
                tokio::time::sleep(backoff).await;
            }
        }
    }
}

/// Stringify a [`FormatType`] for cache-key fingerprinting. Kept
/// local to avoid depending on `Display` on a re-exported enum.
const fn format_type_tag(ft: langextract_core::FormatType) -> &'static str {
    match ft {
        langextract_core::FormatType::Json => "json",
        langextract_core::FormatType::Yaml => "yaml",
    }
}

#[allow(dead_code)]
const _: fn() = || {
    // Compile-time sanity check that ExtractError is still Send —
    // the pipeline crosses .await points with this in its error
    // channel, so losing Send would break concurrent chunks.
    const fn assert_send<T: Send>() {}
    assert_send::<ExtractError>();
};
