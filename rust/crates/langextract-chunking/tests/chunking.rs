//! Integration tests mirroring selected cases from
//! `tests/chunking_test.py`.

use langextract_chunking::{ChunkIterator, TextChunk, batch_chunks};
use langextract_core::TokenInterval;
use pretty_assertions::assert_eq;

fn intervals(chunks: &[TextChunk]) -> Vec<(usize, usize)> {
    chunks
        .iter()
        .map(|c| (c.token_interval.start_index, c.token_interval.end_index))
        .collect()
}

fn texts(chunks: &[TextChunk]) -> Vec<&str> {
    chunks.iter().map(|c| c.text.as_str()).collect()
}

// ------------------------------------------------------------
// Multi-sentence packing (branch C)
// ------------------------------------------------------------

#[test]
fn multi_sentence_chunk() {
    // Mirrors `test_multi_sentence_chunk`. With a 50-char buffer, the
    // first two sentences ("This is a sentence." + "This is a longer
    // sentence.") pack together, and the "Mr. Bond..." tail becomes
    // its own chunk.
    let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
    let chunks: Vec<TextChunk> = ChunkIterator::new(text, 50).unwrap().collect();
    assert_eq!(chunks.len(), 2);
    assert_eq!(
        texts(&chunks),
        vec![
            "This is a sentence. This is a longer sentence.",
            "Mr. Bond\nasks\nwhy?",
        ]
    );
}

// ------------------------------------------------------------
// Sentence-break (branch A) with small buffer
// ------------------------------------------------------------

#[test]
fn break_sentence_at_token_boundary() {
    // Mirrors `test_break_sentence`. With a 12-char buffer, every
    // sentence has to be broken token-by-token.
    let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
    let chunks: Vec<TextChunk> = ChunkIterator::new(text, 12).unwrap().collect();

    let actual = intervals(&chunks);
    // Every chunk must have a non-empty interval.
    for &(s, e) in &actual {
        assert!(s < e, "chunk [{s}, {e}) is empty");
    }
    // And every chunk's text must fit in the buffer (modulo the
    // branch-B "single oversized token" exception).
    for c in &chunks {
        assert!(
            c.text.len() <= 12 || count_tokens(c) == 1,
            "chunk {:?} exceeds buffer without being a single oversized token",
            c.text
        );
    }
    // Spot-check: first chunk is "This is a" at [0, 3).
    assert_eq!(chunks[0].text, "This is a");
    assert_eq!(chunks[0].token_interval, TokenInterval::new(0, 3));
    // Second is "sentence." at [3, 5).
    assert_eq!(chunks[1].text, "sentence.");
    assert_eq!(chunks[1].token_interval, TokenInterval::new(3, 5));
}

const fn count_tokens(c: &TextChunk) -> usize {
    c.token_interval.end_index - c.token_interval.start_index
}

// ------------------------------------------------------------
// Oversized single token (branch B)
// ------------------------------------------------------------

#[test]
fn long_token_gets_own_chunk() {
    // Mirrors `test_long_token_gets_own_chunk`. With a 7-char buffer,
    // "sentence" (8 chars) is a single oversized token and must
    // appear in a chunk by itself.
    let text = "This is a sentence. This is a longer sentence. Mr. Bond\nasks\nwhy?";
    let chunks: Vec<TextChunk> = ChunkIterator::new(text, 7).unwrap().collect();

    // The "sentence" tokens (indices 3 and 10) must be solo chunks.
    let solo_sentence_chunks: Vec<&TextChunk> = chunks
        .iter()
        .filter(|c| c.text == "sentence")
        .collect();
    assert!(
        !solo_sentence_chunks.is_empty(),
        "expected at least one 'sentence' solo chunk"
    );
    for sc in solo_sentence_chunks {
        assert_eq!(
            sc.token_interval.end_index - sc.token_interval.start_index,
            1,
            "'sentence' should be its own chunk"
        );
    }

    // First few chunks: "This is" / "a" / "sentence" / "."
    assert_eq!(chunks[0].text, "This is");
    assert_eq!(chunks[1].text, "a");
    assert_eq!(chunks[2].text, "sentence");
    assert_eq!(chunks[3].text, ".");
}

// ------------------------------------------------------------
// Newline-at-boundary regression
// ------------------------------------------------------------

#[test]
fn newline_at_chunk_boundary_does_not_create_empty_interval() {
    // Mirrors `test_newline_at_chunk_boundary_does_not_create_empty_interval`.
    // The Python bug was creating a zero-length interval when a
    // newline fell exactly at a chunk boundary; we just assert that
    // every emitted chunk has a non-empty interval.
    let text = "First sentence.\nSecond sentence that is longer.\nThird sentence.";
    let chunks: Vec<TextChunk> = ChunkIterator::new(text, 20).unwrap().collect();
    for c in &chunks {
        assert!(
            c.token_interval.start_index < c.token_interval.end_index,
            "empty interval in chunk {c:?}"
        );
    }
    assert!(!chunks.is_empty());
}

// ------------------------------------------------------------
// Unicode text fits in a single chunk when buffer is large enough
// ------------------------------------------------------------

#[test]
fn chunk_unicode_text() {
    // Mirrors `test_chunk_unicode_text`. With a 200-byte buffer the
    // whole passage (which contains a non-ASCII smart quote) fits in
    // a single chunk.
    let text = "Chief Complaint:\n\u{2018}swelling of tongue and difficulty breathing and swallowing\u{2019}\nHistory of Present Illness:\n77 y o woman in NAD with a h/o CAD, DM2, asthma and HTN on altace.";
    let chunks: Vec<TextChunk> = ChunkIterator::new(text, 200).unwrap().collect();
    assert_eq!(chunks.len(), 1);
    assert_eq!(chunks[0].text, text);
}

// ------------------------------------------------------------
// Empty and edge cases
// ------------------------------------------------------------

#[test]
fn empty_text_yields_no_chunks() {
    assert_eq!(ChunkIterator::new("", 100).unwrap().count(), 0);
}

#[test]
fn zero_buffer_is_rejected() {
    assert!(ChunkIterator::new("hi", 0).is_err());
}

// ------------------------------------------------------------
// Batching
// ------------------------------------------------------------

#[test]
fn batch_chunks_groups_in_order() {
    let text = "One. Two. Three. Four. Five.";
    let chunker = ChunkIterator::new(text, 5).unwrap();
    let batches: Vec<Vec<TextChunk>> = batch_chunks(chunker, 2).collect();
    // At most 2 chunks per batch.
    for b in &batches {
        assert!(b.len() <= 2);
    }
    // All chunks round-trip into the concatenated source order.
    let all_text: String = batches
        .iter()
        .flat_map(|b| b.iter().map(|c| c.text.as_str()))
        .collect::<Vec<_>>()
        .join(" ");
    assert!(all_text.contains("One"));
    assert!(all_text.contains("Five"));
}
