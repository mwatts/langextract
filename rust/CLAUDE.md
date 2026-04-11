# Rust port of langextract

This directory contains the Rust port of the [langextract](https://github.com/google/langextract)
library. The Python source lives in the parent directory (`../langextract/`).

## For coding agents

Always load the `rust` skill at the start of each session when working in this project.

## Layout

- `crates/langextract-core/` — pure data types, errors, `LanguageModel` trait, and
  the CLI-provider adapter. Zero dependencies on external LLM SDKs. This is the
  foundational crate every other crate in the workspace depends on.
- `crates/langextract-tokenizer/` — word/number/punctuation tokenizer
  (`RegexTokenizer`), token/sentence-range helpers, and the base data types
  the resolver's fuzzy alignment will build on. Port of
  `langextract/core/tokenizer.py`. Unicode grapheme tokenizer deferred.
- `crates/langextract-aligner/` — two-phase exact + fuzzy token
  alignment of extractions to source text. Port of the `WordAligner`
  half of `langextract/resolver.py`. Uses the `similar` crate for diff.
- `crates/langextract-format/` — JSON/YAML output formatting and
  LLM-output parsing, including fence detection, `<think>` tag
  stripping, wrapper-key handling, top-level list fallback, and
  `extract_ordered_extractions` for turning parsed records into
  `Vec<Extraction>`. Port of `langextract/core/format_handler.py`
  plus the record-walker half of `resolver.py`.
- `crates/langextract-chunking/` — sentence-aware document chunking
  that fits a language model's context window. Three-branch
  algorithm: multi-sentence packing, single long sentence breaking
  at newlines, and oversized-token solo chunks. Port of
  `langextract/chunking.py`.
- `crates/langextract-prompting/` — few-shot prompt construction:
  `PromptTemplateStructured` template type with YAML/JSON file
  loader, `QAPromptGenerator` with Q/A-style rendering, and
  `PromptBuilder` trait with stateless and context-aware
  implementations (cross-chunk context tracking per document).
  Port of `langextract/prompting.py`.

Future crates (not yet ported):
- `crates/langextract-chunking/` — document chunking (`chunking.py`)
- `crates/langextract-pipeline/` — the `extract()` entry point and `annotation.py`
- `crates/langextract-provider-*/` — one crate per LLM backend

## Porting principles

1. **Core crate has no LLM SDK dependencies.** SDK-heavy providers live in their
   own feature-gated or separate crates.
2. **Traits over abstract base classes.** Python `abc.ABC` → Rust trait.
3. **Canonical error structs** (`thiserror`), never `Box<dyn Error>` in public API.
4. **Native async traits with `async_trait`** where dyn-compatibility matters
   (the pipeline holds an `Arc<dyn LanguageModel>`).
5. **Features must be additive** — opting in to `cli-adapter` must never break a
   consumer who doesn't want it.
6. **Document the gotchas in the source**, not just in external docs. The
   `cli_adapter` module is the canonical example of this.

## Commands

```sh
cargo build
cargo test
cargo clippy --all-targets --all-features -- -D warnings
```
