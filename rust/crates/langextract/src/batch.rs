//! Document-level batch runner.
//!
//! [`extract_batch`] runs many [`ExtractRequest`]s in parallel
//! against a shared [`LanguageModel`], honouring a caller-specified
//! document-level concurrency cap. It composes on top of the
//! per-document [`extract_with_report`](crate::extract_with_report)
//! function, which in turn applies chunk-level concurrency from each
//! request's `chunk_concurrency` field.
//!
//! ## Effective concurrency
//!
//! Total in-flight LLM calls = `document_concurrency * chunk_concurrency`.
//!
//! If your provider has a hard rate limit at N concurrent requests,
//! pick values whose product stays under N. A hosted API with a
//! budget of 50 concurrent: `document_concurrency=5,
//! chunk_concurrency=10`. A CLI agent at 2 concurrent:
//! `document_concurrency=2, chunk_concurrency=1`.
//!
//! ## Checkpointing
//!
//! Pass a [`Checkpoint`] implementation to skip documents that have
//! already completed in a previous run. The batch runner consults the
//! checkpoint before calling `extract_with_report` for each request
//! and marks the document id as completed on success. Failures do
//! **not** mark the checkpoint — a re-run will retry them.
//!
//! ## Results
//!
//! [`extract_batch`] returns a `Vec<BatchItem>` in input order. Each
//! `BatchItem` is either:
//!
//! - a successful `(AnnotatedDocument, DocumentReport)` pair,
//! - a failure with the [`ExtractError`] that killed the doc,
//! - or a `Skipped { id }` marker for documents the checkpoint said
//!   were already done.
//!
//! Failed documents do not abort the whole batch.

use std::sync::Arc;

use futures::StreamExt;
use futures::stream;
use langextract_core::{AnnotatedDocument, LanguageModel};
use tracing::{Instrument, info, info_span, warn};

use crate::checkpoint::{Checkpoint, NoOpCheckpoint};
use crate::error::ExtractError;
use crate::pipeline::extract_with_report;
use crate::report::DocumentReport;
use crate::request::ExtractRequest;

/// Options for a batch run.
#[derive(Debug, Clone)]
pub struct BatchOptions {
    /// Number of documents to process concurrently. Combines with
    /// each request's `chunk_concurrency` for total in-flight LLM
    /// calls. Default `1`.
    pub document_concurrency: usize,

    /// Checkpoint backend. Default [`NoOpCheckpoint`].
    pub checkpoint: Arc<dyn Checkpoint>,
}

impl Default for BatchOptions {
    fn default() -> Self {
        Self {
            document_concurrency: 1,
            checkpoint: Arc::new(NoOpCheckpoint),
        }
    }
}

/// One entry in a batch result.
#[expect(
    clippy::large_enum_variant,
    reason = "the Completed variant naturally carries the full AnnotatedDocument + report; \
              boxing would force callers to dereference on every access"
)]
#[derive(Debug)]
pub enum BatchItem {
    /// Document processed successfully.
    Completed {
        /// Document id.
        id: String,
        /// The grounded document.
        document: AnnotatedDocument,
        /// Observability report.
        report: DocumentReport,
    },
    /// Document was skipped because the checkpoint marked it done.
    Skipped {
        /// Document id.
        id: String,
    },
    /// Document failed.
    Failed {
        /// Document id.
        id: String,
        /// The error that killed the run.
        error: ExtractError,
    },
}

impl BatchItem {
    /// Returns `true` for [`BatchItem::Completed`].
    #[must_use]
    pub const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// Returns the document id regardless of variant.
    #[must_use]
    pub fn id(&self) -> &str {
        match self {
            Self::Completed { id, .. } | Self::Skipped { id } | Self::Failed { id, .. } => id,
        }
    }
}

/// Run a batch of [`ExtractRequest`]s against a shared language model.
///
/// Returns one [`BatchItem`] per input in input order. A failure on
/// any single document does **not** abort the batch — the item is
/// marked [`BatchItem::Failed`] and processing continues.
///
/// # Errors
///
/// This function itself never returns an error — every per-document
/// failure is embedded in the returned `BatchItem`. The only way for
/// it to fail is a programming error (panic), which propagates
/// naturally.
pub async fn extract_batch(
    model: &dyn LanguageModel,
    requests: Vec<ExtractRequest>,
    options: BatchOptions,
) -> Vec<BatchItem> {
    let concurrency = options.document_concurrency.max(1);
    let checkpoint = options.checkpoint.clone();

    let total = requests.len();
    let batch_span = info_span!(
        "langextract.batch",
        total = total,
        doc_concurrency = concurrency,
    );
    let _enter = batch_span.enter();
    info!(total, concurrency, "starting batch run");

    // Pre-compute the id each request will use so we can honour the
    // checkpoint without moving the request into the task. We
    // materialise into a Vec here because we then move `requests`
    // into the job builder on the next line.
    let ids: Vec<String> = requests
        .iter()
        .map(|r| {
            r.document_id
                .as_ref()
                .map(|id| id.as_str().to_owned())
                .unwrap_or_default()
        })
        .collect();

    let jobs = requests
        .into_iter()
        .zip(ids)
        .enumerate()
        .map(|(idx, (request, precomputed_id))| {
            let checkpoint = Arc::clone(&checkpoint);
            let model_ref: &dyn LanguageModel = model;
            async move {
                // Determine the checkpoint id. If the request has an
                // explicit document_id, use it; otherwise we can't
                // dedup (the pipeline will mint a random id and each
                // run will be fresh).
                let cp_id = if precomputed_id.is_empty() {
                    format!("batch_idx_{idx}")
                } else {
                    precomputed_id.clone()
                };
                if checkpoint.is_completed(&cp_id) {
                    info!(id = %cp_id, "skipping: already completed");
                    return BatchItem::Skipped { id: cp_id };
                }

                let doc_span = info_span!("langextract.batch.item", id = %cp_id);
                let result = async {
                    extract_with_report(model_ref, request).await
                }
                .instrument(doc_span)
                .await;

                match result {
                    Ok((document, report)) => {
                        if let Err(e) = checkpoint.mark_completed(&cp_id) {
                            warn!(id = %cp_id, err = %e, "checkpoint mark_completed failed");
                        }
                        BatchItem::Completed {
                            id: cp_id,
                            document,
                            report,
                        }
                    }
                    Err(error) => {
                        warn!(id = %cp_id, %error, "document failed");
                        BatchItem::Failed { id: cp_id, error }
                    }
                }
            }
        });

    let mut stream = stream::iter(jobs).buffered(concurrency);
    let mut out = Vec::with_capacity(total);
    while let Some(item) = stream.next().await {
        out.push(item);
    }
    info!(
        completed = out.iter().filter(|i| i.is_completed()).count(),
        failed = out.iter().filter(|i| matches!(i, BatchItem::Failed { .. })).count(),
        skipped = out.iter().filter(|i| matches!(i, BatchItem::Skipped { .. })).count(),
        "batch run complete",
    );
    out
}
