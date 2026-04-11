//! End-to-end demo: run the langextract pipeline on the **full** IRS
//! 2025 1040 Instructions PDF (~1 MB of text, ~250 chunks) and
//! report grounded extractions across every page.
//!
//! ## What this demonstrates
//!
//! - Chunking a nearly-1 MB document into ~250 sentence-aware
//!   chunks with [`ChunkIterator`](langextract::ChunkIterator).
//! - Streaming each chunk through a [`LanguageModel`] implementation.
//! - Parsing fenced JSON back out of each chunk's response, walking
//!   records into [`Extraction`]s, and aligning them to the chunk
//!   text with **absolute** source offsets (not chunk-relative).
//! - Merging every chunk's grounded extractions into a single
//!   [`AnnotatedDocument`] and reporting class-level and
//!   per-chunk statistics.
//!
//! ## The "LLM"
//!
//! Processing a 250-chunk document through a hosted LLM or a
//! coding-agent CLI would take minutes or hours and burn real tokens.
//! For a reproducible, deterministic demo we instead plug in a
//! [`RegexMinedModel`] — a tiny `LanguageModel` implementation that
//! scans each chunk for well-known IRS entity patterns (form
//! references, publications, schedules, tax topics, tax credits,
//! dollar amounts, percentages, tax years) via compiled regexes and
//! emits them as fenced JSON that looks exactly like what a real LLM
//! would produce.
//!
//! The point is **not** that regexes are a good extraction strategy
//! — they're brittle, locale-specific, and miss anything novel. The
//! point is that langextract's pipeline treats every
//! `LanguageModel` uniformly: parse → walk → align → ground. Swap
//! `RegexMinedModel` for a
//! [`CliLanguageModel`](langextract_core::cli_adapter::CliLanguageModel)
//! wrapping a real CLI agent and the rest of the code is
//! byte-for-byte identical.
//!
//! ## Running
//!
//! ```sh
//! # 1. Download and extract the PDF once.
//! mkdir -p /tmp/langextract-demo
//! curl -sSL -o /tmp/langextract-demo/i1040gi.pdf \
//!   https://www.irs.gov/pub/irs-pdf/i1040gi.pdf
//! pdftotext -layout /tmp/langextract-demo/i1040gi.pdf \
//!           /tmp/langextract-demo/full.txt
//!
//! # 2. Run the demo.
//! cargo run --example irs_1040_demo --release
//! ```
//!
//! The default path is `/tmp/langextract-demo/full.txt`; override
//! with `LANGEXTRACT_DEMO_TEXT_PATH=/some/other/path`.

use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use async_trait::async_trait;
use langextract::{
    AlignmentStatus, ChunkCache, DocumentHealthThresholds, ExampleData, ExtractRequest,
    Extraction, FormatType, InMemoryChunkCache, InferError, InferenceParams, LanguageModel,
    RetryPolicy, ScoredOutput, extract_with_report,
};
use regex::Regex;

// ------------------------------------------------------------
// Prompt markers
// ------------------------------------------------------------
//
// `QAPromptGenerator` uses `"Q: "` and `"A: "` as the default line
// prefixes. The rendered prompt ends with the current chunk wrapped
// in one last Q:/A: pair:
//
//     ...previous examples...
//
//     Q: <chunk text>
//     A:
//
// The chunk text lives between the *last* `"Q: "` and the final
// trailing `"\nA: "` line. Using `rfind` here skips every example's
// Q: line correctly because the chunk's Q: is the last one in the
// prompt.

const Q_MARKER: &str = "Q: ";
const A_MARKER: &str = "\nA: ";

/// Pull the chunk text out of a rendered prompt.
fn extract_chunk_from_prompt(prompt: &str) -> &str {
    let Some(q_pos) = prompt.rfind(Q_MARKER) else {
        return "";
    };
    let after_q = &prompt[q_pos + Q_MARKER.len()..];
    let Some(a_rel) = after_q.rfind(A_MARKER) else {
        return after_q.trim();
    };
    after_q[..a_rel].trim()
}

// ------------------------------------------------------------
// Regex-based entity miner
// ------------------------------------------------------------

/// Captures a `(class, text)` pair for every match a regex finds.
fn mine_with(re: &Regex, class: &str, chunk: &str, out: &mut Vec<(String, String)>) {
    for m in re.find_iter(chunk) {
        out.push((class.to_owned(), m.as_str().to_owned()));
    }
}

/// Scan `chunk` for IRS-style entities and return them as
/// `(class, text)` pairs in source order.
fn mine_chunk(chunk: &str) -> Vec<(String, String)> {
    // Compile once per call — a production version would cache with
    // `OnceLock`, but with ~250 chunks the cost is noise.
    let form_re = Regex::new(r"\bForm\s+\d{3,5}(?:-[A-Z0-9]{1,3})?(?:-SR)?\b").unwrap();
    let w_form_re = Regex::new(r"\bForm\s+W-\d{1,2}[A-Z]?\b").unwrap();
    let pub_re = Regex::new(r"\bPub\.\s*\d{1,4}[A-Z]?\b").unwrap();
    let schedule_re = Regex::new(r"\bSchedule\s+[A-Z0-9]{1,3}(?:-[A-Z0-9]+)?\b").unwrap();
    let topic_re = Regex::new(r"\bTax\s+Topic\s+\d{2,4}\b").unwrap();
    let tax_year_re = Regex::new(r"\b20\d{2}\b").unwrap();
    // Dollar amounts: "$1,234", "$1,234.56", "$50"
    let money_re = Regex::new(r"\$\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?\b").unwrap();
    let percent_re = Regex::new(r"\b\d{1,3}(?:\.\d+)?%").unwrap();

    // Tax credits: a handful of well-known names. Match
    // case-insensitively and store in the canonical capitalization
    // the source happens to use.
    let credit_re = Regex::new(
        r"(?i)\b(?:earned\s+income\s+credit|additional\s+child\s+tax\s+credit|\
             child\s+tax\s+credit|american\s+opportunity\s+credit|\
             premium\s+tax\s+credit|refundable\s+adoption\s+credit|\
             saver['\u2019]s\s+credit|lifetime\s+learning\s+credit|\
             credit\s+for\s+other\s+dependents)\b",
    )
    .unwrap();

    let mut out: Vec<(String, String)> = Vec::new();
    mine_with(&form_re, "form_reference", chunk, &mut out);
    mine_with(&w_form_re, "form_reference", chunk, &mut out);
    mine_with(&pub_re, "publication", chunk, &mut out);
    mine_with(&schedule_re, "schedule", chunk, &mut out);
    mine_with(&topic_re, "tax_topic", chunk, &mut out);
    mine_with(&money_re, "dollar_amount", chunk, &mut out);
    mine_with(&percent_re, "percentage", chunk, &mut out);
    mine_with(&credit_re, "tax_credit", chunk, &mut out);

    // Tax years: only keep 2020..=2030 to skip stray 4-digit numbers.
    for m in tax_year_re.find_iter(chunk) {
        let s = m.as_str();
        if let Ok(y) = s.parse::<u32>() {
            if (2020..=2030).contains(&y) {
                out.push(("tax_year".to_owned(), s.to_owned()));
            }
        }
    }

    out
}

/// Escape a string for safe embedding inside a JSON string literal.
/// Handles the standard metacharacters plus ASCII control characters
/// (which pdftotext occasionally emits as layout hints).
fn json_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len() + 2);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '"' => out.push_str("\\\""),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\x08' => out.push_str("\\b"),
            '\x0c' => out.push_str("\\f"),
            c if (c as u32) < 0x20 => {
                use std::fmt::Write;
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out
}

/// Build the fenced-JSON response a real LLM would return for a
/// chunk, given the pre-computed `(class, text)` pairs. Uses the
/// default `_attributes` and `_index` suffixes so the pipeline can
/// order by index and attach empty attribute objects.
fn render_json_response(pairs: &[(String, String)]) -> String {
    use std::fmt::Write as _;
    let mut records = String::new();
    records.push_str("[\n");
    for (i, (class, text)) in pairs.iter().enumerate() {
        if i > 0 {
            records.push_str(",\n");
        }
        let escaped = json_escape(text);
        let class_escaped = json_escape(class);
        let idx = i + 1;
        let _ = write!(
            records,
            "    {{\"{class_escaped}\": \"{escaped}\", \"{class_escaped}_index\": {idx}, \"{class_escaped}_attributes\": {{}}}}"
        );
    }
    records.push_str("\n  ]");

    format!("```json\n{{\n  \"extractions\": {records}\n}}\n```")
}

/// A deterministic `LanguageModel` that mines each chunk's prompt
/// for IRS-style entities and emits a fenced JSON response. Plays
/// the same role a hosted LLM or a CLI agent would.
#[derive(Debug)]
struct RegexMinedModel {
    // Simple statistics — kept behind a Mutex so we can record per
    // chunk and dump a summary at the end without fighting the
    // borrow checker.
    call_count: Mutex<usize>,
    per_class_counts: Mutex<BTreeMap<String, usize>>,
}

impl RegexMinedModel {
    const fn new() -> Self {
        Self {
            call_count: Mutex::new(0),
            per_class_counts: Mutex::new(BTreeMap::new()),
        }
    }

    fn snapshot(&self) -> (usize, BTreeMap<String, usize>) {
        let calls = *self.call_count.lock().unwrap();
        let classes = self.per_class_counts.lock().unwrap().clone();
        (calls, classes)
    }
}

#[async_trait]
impl LanguageModel for RegexMinedModel {
    async fn infer(
        &self,
        prompts: &[String],
        _params: &InferenceParams,
    ) -> Result<Vec<Vec<ScoredOutput>>, InferError> {
        let mut out = Vec::with_capacity(prompts.len());
        for prompt in prompts {
            {
                let mut count = self.call_count.lock().unwrap();
                *count += 1;
            }
            let chunk = extract_chunk_from_prompt(prompt);
            let pairs = mine_chunk(chunk);

            {
                let mut classes = self.per_class_counts.lock().unwrap();
                for (class, _) in &pairs {
                    *classes.entry(class.clone()).or_insert(0) += 1;
                }
            }

            out.push(vec![ScoredOutput::unscored(render_json_response(&pairs))]);
        }
        Ok(out)
    }
}

// ------------------------------------------------------------
// main
// ------------------------------------------------------------

#[tokio::main(flavor = "current_thread")]
#[expect(
    clippy::too_many_lines,
    reason = "demo main is an illustrative walkthrough; splitting hides the sequence of steps"
)]
async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let path = std::env::var("LANGEXTRACT_DEMO_TEXT_PATH")
        .unwrap_or_else(|_| "/tmp/langextract-demo/full.txt".to_owned());

    println!("=== langextract / full IRS 1040 Instructions demo ===\n");
    println!("Reading text from: {path}");

    let source = std::fs::read_to_string(&path).map_err(|e| {
        format!("failed to read {path}: {e}\n\nRun the download step from the example docs first.")
    })?;

    println!("Source length:     {} bytes", source.len());
    println!("Source lines:      {}", source.lines().count());
    println!();

    let model = RegexMinedModel::new();

    // Share a chunk cache across a hypothetical batch so the
    // second document pays zero LLM cost for boilerplate chunks.
    // For this single-doc demo, we still install it to demonstrate
    // the plumbing.
    let cache: Arc<InMemoryChunkCache> = Arc::new(InMemoryChunkCache::new());

    let request = ExtractRequest {
        text: source.clone(),
        description:
            "Extract all IRS form references, publications, schedules, tax topics, \
             tax credits, dollar amounts, percentages, and tax years from the \
             following document text. Preserve order of appearance."
                .to_owned(),
        examples: vec![ExampleData::new(
            "See Pub. 17 and Schedule A on Form 1040 for details.",
            vec![
                Extraction::new("publication", "Pub. 17"),
                Extraction::new("schedule", "Schedule A"),
                Extraction::new("form_reference", "Form 1040"),
            ],
        )],
        format_type: FormatType::Json,
        max_char_buffer: 4000, // ~250 chunks over 1 MB of text
        extraction_index_suffix: Some("_index".to_owned()),
        // Kept ENABLED — the Tier-1 fuzzy safeguards (see
        // FuzzySafeguards::default) cap per-chunk fuzzy work so the
        // regex miner's dense output can't DoS the pipeline the way
        // it did before the safeguards landed.
        enable_fuzzy_alignment: true,
        // Chunk concurrency: run 4 chunks in parallel. For a
        // scripted/regex-based model this costs nothing and
        // exercises the pipeline's concurrent path.
        chunk_concurrency: 4,
        // Retry policy: none here because the scripted model never
        // fails. Real providers should leave this at the default.
        retry_policy: RetryPolicy::none(),
        // Shared content-addressed cache.
        chunk_cache: cache.clone() as Arc<dyn langextract::ChunkCache>,
        ..Default::default()
    };

    let start = Instant::now();
    let (result, report) = extract_with_report(&model, request).await?;
    let elapsed = start.elapsed();

    println!("=== pipeline run complete ===");
    println!("Wall time:         {elapsed:.2?}");
    println!("Pipeline elapsed:  {:.2?}", report.elapsed);

    let (call_count, per_class_emitted) = model.snapshot();
    println!("Model calls:       {call_count}");
    println!("Chunks in report:  {}", report.chunk_count());
    println!("Extractions total: {}", result.extractions.len());

    // Tier-1 observability: cache behaviour + retries.
    println!("Cache hits:        {}", report.cache_hits);
    println!("Cache misses:      {}", report.cache_misses);
    println!("Cache entries:     {:?}", cache.len());
    println!("Total retries:     {}", report.total_retries);

    // Tier-2 health gate: is this document safe to promote into the
    // downstream rule-extraction layer?
    let thresholds = DocumentHealthThresholds::default();
    let health = report.health(&thresholds);
    println!("Health status:     {health:?}");
    println!("Grounding rate:    {:.1}%", report.grounding_rate() * 100.0);
    println!("Fuzzy fraction:    {:.1}%", report.fuzzy_fraction() * 100.0);

    // Tier-2 unaligned-reason diagnostics.
    if !report.unaligned_reason_histogram.is_empty() {
        println!("\n─── Unaligned reasons ───");
        let mut sorted: Vec<(&&str, &usize)> =
            report.unaligned_reason_histogram.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1));
        for (reason, count) in sorted {
            println!("  {count:>5}×  {reason}");
        }
    }

    let aligned_count = result
        .extractions
        .iter()
        .filter(|e| e.char_interval.is_some())
        .count();
    let unaligned_count = result.extractions.len() - aligned_count;
    let exact_count = result
        .extractions
        .iter()
        .filter(|e| matches!(e.alignment_status, Some(AlignmentStatus::MatchExact)))
        .count();
    let fuzzy_count = result
        .extractions
        .iter()
        .filter(|e| matches!(e.alignment_status, Some(AlignmentStatus::MatchFuzzy)))
        .count();
    let lesser_count = result
        .extractions
        .iter()
        .filter(|e| matches!(e.alignment_status, Some(AlignmentStatus::MatchLesser)))
        .count();

    println!("  grounded:        {aligned_count}");
    println!("  unaligned:       {unaligned_count}");
    println!("  exact matches:   {exact_count}");
    println!("  fuzzy matches:   {fuzzy_count}");
    println!("  lesser matches:  {lesser_count}");

    // Group by class and show counts.
    let mut grounded_per_class: BTreeMap<&str, usize> = BTreeMap::new();
    for ex in &result.extractions {
        if ex.char_interval.is_some() {
            *grounded_per_class
                .entry(ex.extraction_class.as_str())
                .or_insert(0) += 1;
        }
    }

    println!("\n─── Grounded extractions by class ───");
    println!("{:<16}  emitted   grounded", "class");
    println!("{:<16}  -------   --------", "-----");
    for class in per_class_emitted.keys() {
        let emitted = per_class_emitted.get(class).copied().unwrap_or(0);
        let grounded = grounded_per_class.get(class.as_str()).copied().unwrap_or(0);
        println!("{class:<16}  {emitted:>7}   {grounded:>8}");
    }

    // Show the top-N most frequently grounded extractions (verbatim
    // text occurrences) per class — a poor-man's "topics in the
    // document".
    println!("\n─── Top extractions per class (top 5 each) ───");
    let mut per_class_texts: BTreeMap<&str, BTreeMap<&str, usize>> = BTreeMap::new();
    for ex in &result.extractions {
        if ex.char_interval.is_none() {
            continue;
        }
        *per_class_texts
            .entry(ex.extraction_class.as_str())
            .or_default()
            .entry(ex.extraction_text.as_str())
            .or_insert(0) += 1;
    }
    for (class, texts) in &per_class_texts {
        println!("\n[{class}]");
        let mut sorted: Vec<(&&str, &usize)> = texts.iter().collect();
        sorted.sort_by(|a, b| b.1.cmp(a.1).then_with(|| a.0.cmp(b.0)));
        for (text, count) in sorted.iter().take(5) {
            println!("  {count:>4}×  {text}");
        }
    }

    // Sanity check: for every grounded extraction, the source slice
    // at char_interval must be consistent with the alignment status.
    //
    // - MATCH_EXACT: slice must equal the extraction text byte-for-byte.
    // - MATCH_LESSER: slice is a *subset* of the extraction — the
    //   extraction has more tokens than actually appear in the
    //   source. The slice should be non-empty and should share at
    //   least some token with the extraction.
    //
    // Unaligned extractions have no char_interval to check.
    let mut exact_mismatches = 0;
    let mut empty_slice_count = 0;
    for ex in &result.extractions {
        let Some(ci) = ex.char_interval else { continue };
        let slice = &source[ci.start..ci.end];
        match ex.alignment_status {
            Some(AlignmentStatus::MatchExact) => {
                if slice != ex.extraction_text {
                    exact_mismatches += 1;
                }
            }
            _ => {
                if slice.is_empty() {
                    empty_slice_count += 1;
                }
            }
        }
    }
    println!("\n─── Grounding sanity check ───");
    println!("MATCH_EXACT extractions with slice != text:  {exact_mismatches}");
    println!("MATCH_LESSER extractions with empty slice:   {empty_slice_count}");
    println!(
        "(Non-zero MATCH_LESSER slice lengths are expected: \
         they're valid partial-match spans, not bugs.)"
    );

    println!(
        "\nProcessed {} bytes across {} chunks in {:.2?} → \
         {} grounded extractions.",
        source.len(),
        call_count,
        elapsed,
        aligned_count,
    );

    Ok(())
}
