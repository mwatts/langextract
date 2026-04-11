//! The [`ExtractRequest`] struct — everything needed to run a
//! single end-to-end extraction against a language model.

use std::sync::Arc;

use langextract_aligner::{DEFAULT_FUZZY_THRESHOLD, FuzzySafeguards};
use langextract_core::{ATTRIBUTE_SUFFIX, DocumentId, ExampleData, FormatType};

use crate::cache::{ChunkCache, NoOpChunkCache};
use crate::retry::RetryPolicy;

/// Default chunk size in bytes.
///
/// Chosen conservatively to fit almost any model's context window
/// while leaving room for the prompt itself. Callers with
/// big-context models (Gemini, Claude 3/4, modern `OpenAI`
/// long-context variants) can raise this substantially.
pub const DEFAULT_MAX_CHAR_BUFFER: usize = 1500;

/// Default index-field suffix used by [`ExtractRequest`] when
/// `extraction_index_suffix = Some(...)`. Absent by default —
/// auto-incrementing ordering is usually what callers want.
pub const DEFAULT_INDEX_SUFFIX: &str = "_index";

/// A single end-to-end extraction request.
///
/// Construct via a struct literal, or start from
/// `ExtractRequest::new(text, description)` and override individual
/// fields. The type implements [`Default`] so you can use the
/// `..Default::default()` shorthand for the knobs you don't care
/// about.
///
/// ```
/// use langextract::ExtractRequest;
/// use langextract_core::Extraction;
///
/// # fn _example() {
/// let request = ExtractRequest {
///     text: "Alice is an engineer at Acme Corp.".to_owned(),
///     description: "Extract people and their roles.".to_owned(),
///     examples: vec![/* ExampleData... */],
///     max_char_buffer: 4000,
///     ..Default::default()
/// };
/// # drop(request);
/// # }
/// ```
#[expect(
    clippy::struct_excessive_bools,
    reason = "each flag is an orthogonal Python-parity knob; packing them into \
              a state machine would obscure the 1:1 mapping to FormatHandler / \
              AlignmentOptions / ContextAwarePromptBuilder that this request struct is a facade over"
)]
#[derive(Debug, Clone)]
pub struct ExtractRequest {
    /// The source document to extract from.
    pub text: String,

    /// Natural-language description of the extraction task, used as
    /// the top of the few-shot prompt.
    pub description: String,

    /// Few-shot examples illustrating the expected schema.
    pub examples: Vec<ExampleData>,

    /// Maximum chunk size in bytes. The chunker splits the source at
    /// sentence boundaries so each piece fits this budget.
    pub max_char_buffer: usize,

    /// Output format the model is prompted to produce. `Json` by
    /// default (matches the overwhelming majority of providers).
    pub format_type: FormatType,

    /// Whether to wrap the extractions list in a container object
    /// (`{"extractions": [...]}`). Defaults to `true`.
    pub use_wrapper: bool,

    /// Wrapper key. Defaults to `"extractions"`.
    pub wrapper_key: String,

    /// Attribute-field suffix (`"_attributes"` by default). Must
    /// match the suffix the model will use in its output.
    pub attribute_suffix: String,

    /// Extraction-index-field suffix. `None` (the default) means the
    /// pipeline auto-increments indices instead of reading them from
    /// the model output. `Some("_index")` makes the pipeline read
    /// `<class>_index` fields and drop any extraction whose class
    /// has no matching index.
    pub extraction_index_suffix: Option<String>,

    /// Whether to use Markdown fences (```json … ```) around formatted
    /// examples and when parsing model output. Defaults to `true`.
    pub use_fences: bool,

    /// Whether to enable per-extraction sliding-window fuzzy
    /// alignment when exact matching fails.
    pub enable_fuzzy_alignment: bool,

    /// Minimum match ratio for fuzzy alignment, 0.0..=1.0.
    pub fuzzy_alignment_threshold: f32,

    /// Whether to classify partial exact matches as
    /// [`AlignmentStatus::MatchLesser`](langextract_core::AlignmentStatus::MatchLesser).
    pub accept_match_lesser: bool,

    /// Byte window of previous-chunk text to include as cross-chunk
    /// context. `None` (default) disables cross-chunk context; use
    /// `Some(100..500)` for models that need help resolving
    /// coreferences across chunk boundaries.
    pub context_window_chars: Option<usize>,

    /// Optional document ID. Used to isolate cross-chunk context
    /// state when running multiple documents through the same
    /// pipeline, and propagated onto the final `AnnotatedDocument`.
    pub document_id: Option<DocumentId>,

    /// Optional additional context to inject into every prompt
    /// (e.g. "This is a clinical note from a radiology report.").
    pub additional_context: Option<String>,

    /// Number of chunks from this document to process concurrently.
    /// The pipeline uses `futures::stream::buffer_unordered(N)` to
    /// drive up to `N` per-chunk tasks in parallel. Default `1`
    /// (serial). Raise to match your provider's concurrency budget
    /// — 1-2 for a heavy CLI agent, 10-50 for a hosted API.
    pub chunk_concurrency: usize,

    /// Retry policy for per-chunk LLM calls. Transient errors
    /// (rate limits, timeouts, malformed responses) trigger a
    /// retry; fatal errors propagate immediately.
    pub retry_policy: RetryPolicy,

    /// Chunk-level response cache. Default: [`NoOpChunkCache`].
    /// Supply an [`InMemoryChunkCache`](crate::cache::InMemoryChunkCache)
    /// to share a cache across runs in a batch, or implement the
    /// [`ChunkCache`] trait for persistent / distributed backends.
    pub chunk_cache: Arc<dyn ChunkCache>,

    /// Safeguards applied to the fuzzy alignment phase. See
    /// [`FuzzySafeguards`].
    pub fuzzy_safeguards: FuzzySafeguards,
}

impl ExtractRequest {
    /// Construct a request with the two required fields. All other
    /// fields are initialised from [`Default`].
    #[must_use]
    pub fn new(text: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            text: text.into(),
            description: description.into(),
            ..Self::default()
        }
    }
}

impl Default for ExtractRequest {
    fn default() -> Self {
        Self {
            text: String::new(),
            description: String::new(),
            examples: Vec::new(),
            max_char_buffer: DEFAULT_MAX_CHAR_BUFFER,
            format_type: FormatType::Json,
            use_wrapper: true,
            wrapper_key: "extractions".to_owned(),
            attribute_suffix: ATTRIBUTE_SUFFIX.to_owned(),
            extraction_index_suffix: None,
            use_fences: true,
            enable_fuzzy_alignment: true,
            fuzzy_alignment_threshold: DEFAULT_FUZZY_THRESHOLD,
            accept_match_lesser: true,
            context_window_chars: None,
            document_id: None,
            additional_context: None,
            chunk_concurrency: 1,
            retry_policy: RetryPolicy::default(),
            chunk_cache: Arc::new(NoOpChunkCache),
            fuzzy_safeguards: FuzzySafeguards::default(),
        }
    }
}
