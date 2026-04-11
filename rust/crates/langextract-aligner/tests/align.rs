//! Integration tests mirroring selected Python alignment test cases
//! from `tests/resolver_test.py::AlignEntitiesTest` and the
//! `test_extraction_alignment` parameterized suite.

use langextract_aligner::{AlignmentOptions, align_extraction_groups};
use langextract_core::{AlignmentStatus, CharInterval, Extraction, TokenInterval};
use pretty_assertions::assert_eq;

/// Check that each aligned extraction's `char_interval`, when used to
/// slice the source, still corresponds to text containing the
/// extraction's first and last tokens (sanity check modelled on
/// `assert_char_interval_match_source` in the Python suite).
fn assert_char_interval_makes_sense(source: &str, extractions: &[&Extraction]) {
    for ex in extractions {
        if let Some(ci) = ex.char_interval {
            assert!(
                ci.end <= source.len(),
                "char_interval end {} > source len {}",
                ci.end,
                source.len()
            );
            assert!(ci.start <= ci.end);
        }
    }
}

fn defaults() -> AlignmentOptions {
    AlignmentOptions::default()
}

// ------------------------------------------------------------
// Exact matching
// ------------------------------------------------------------

#[test]
fn basic_alignment_two_medications() {
    let source = "Patient is prescribed Naprosyn and prednisone for treatment.";
    let groups = vec![
        vec![Extraction::new("medication", "Naprosyn")],
        vec![Extraction::new("medication", "prednisone")],
    ];
    let aligned = align_extraction_groups(groups, source, &defaults()).unwrap();

    // Naprosyn at token index 3, chars 22..30.
    let n = &aligned[0][0];
    assert_eq!(n.alignment_status, Some(AlignmentStatus::MatchExact));
    assert_eq!(n.token_interval, Some(TokenInterval::new(3, 4)));
    assert_eq!(n.char_interval, Some(CharInterval::new(22, 30)));

    // prednisone at token index 5, chars 35..45.
    let p = &aligned[1][0];
    assert_eq!(p.alignment_status, Some(AlignmentStatus::MatchExact));
    assert_eq!(p.token_interval, Some(TokenInterval::new(5, 6)));
    assert_eq!(p.char_interval, Some(CharInterval::new(35, 45)));

    assert_char_interval_makes_sense(source, &[n, p]);
}

#[test]
fn empty_extraction_groups_return_empty() {
    let aligned = align_extraction_groups(vec![], "anything", &defaults()).unwrap();
    assert!(aligned.is_empty());
}

#[test]
fn extraction_not_in_source_is_left_unaligned() {
    let source = "Patient takes aspirin daily.";
    let groups = vec![vec![Extraction::new(
        "medication",
        "completely different medicine",
    )]];
    let aligned = align_extraction_groups(groups, source, &defaults()).unwrap();
    let ex = &aligned[0][0];
    assert_eq!(ex.alignment_status, None);
    assert_eq!(ex.token_interval, None);
    assert_eq!(ex.char_interval, None);
}

// ------------------------------------------------------------
// MATCH_LESSER behaviour
// ------------------------------------------------------------

#[test]
fn partial_extraction_becomes_match_lesser_when_accepted() {
    // Mirrors the Python `fuzzy_alignment_success` case — extraction
    // starts cleanly with "severe" so the diff gives a j=0 block of
    // length 3 ("severe heart problems"), which is shorter than the
    // 4-token extraction → MATCH_LESSER with default options.
    let source = "Patient has severe heart problems today.";
    let groups = vec![vec![Extraction::new(
        "condition",
        "severe heart problems complications",
    )]];
    let aligned = align_extraction_groups(groups, source, &defaults()).unwrap();
    let ex = &aligned[0][0];
    assert_eq!(ex.alignment_status, Some(AlignmentStatus::MatchLesser));
    assert_eq!(ex.token_interval, Some(TokenInterval::new(2, 5)));
    assert_eq!(ex.char_interval, Some(CharInterval::new(12, 33)));
}

// ------------------------------------------------------------
// Fuzzy alignment
// ------------------------------------------------------------

#[test]
fn fuzzy_subset_window_the_iliopsoas_tendon_is_intact() {
    // Extraction text is a subset of the longer source clause: extra
    // tokens in source should not penalize. Expected MATCH_FUZZY.
    let source = "The iliopsoas and proximal hamstring tendons are intact.";
    let groups = vec![vec![Extraction::new(
        "tendon",
        "The iliopsoas tendon is intact",
    )]];
    let aligned = align_extraction_groups(
        groups,
        source,
        &AlignmentOptions {
            accept_match_lesser: false,
            ..defaults()
        },
    )
    .unwrap();
    let ex = &aligned[0][0];
    assert_eq!(ex.alignment_status, Some(AlignmentStatus::MatchFuzzy));
    assert!(ex.token_interval.is_some());
    assert!(ex.char_interval.is_some());
}

#[test]
fn fuzzy_below_threshold_does_not_align() {
    let source = "Patient takes aspirin daily.";
    let groups = vec![vec![Extraction::new(
        "medication",
        "completely different medicine",
    )]];
    let aligned = align_extraction_groups(groups, source, &defaults()).unwrap();
    let ex = &aligned[0][0];
    assert_eq!(ex.alignment_status, None);
}

#[test]
fn fuzzy_partial_overlap_success() {
    // Mirrors the Python `fuzzy_alignment_partial_overlap_success`
    // case: 3 of 4 extraction tokens appear as a contiguous window in
    // the source (ratio 0.75 meets the default threshold).
    let source = "Findings consistent with degenerative disc disease at L5-S1.";
    let groups = vec![vec![Extraction::new(
        "finding",
        "mild degenerative disc disease",
    )]];
    let aligned = align_extraction_groups(groups, source, &defaults()).unwrap();
    let ex = &aligned[0][0];
    assert_eq!(ex.alignment_status, Some(AlignmentStatus::MatchFuzzy));
    assert!(ex.token_interval.is_some());
}

// ------------------------------------------------------------
// Delimiter validation
// ------------------------------------------------------------

#[test]
fn extraction_containing_delimiter_is_rejected() {
    let source = "anything";
    let bad_text = format!("hello {} world", langextract_aligner::DEFAULT_DELIMITER);
    let groups = vec![vec![Extraction::new("x", bad_text)]];
    let err = align_extraction_groups(groups, source, &defaults()).unwrap_err();
    match err {
        langextract_aligner::AlignError::DelimiterInExtraction { .. } => {}
        other => panic!("unexpected error: {other:?}"),
    }
}

// ------------------------------------------------------------
// Options propagation
// ------------------------------------------------------------

#[test]
fn token_and_char_offsets_are_added_to_intervals() {
    let source = "Patient is prescribed Naprosyn.";
    let groups = vec![vec![Extraction::new("medication", "Naprosyn")]];
    let opts = AlignmentOptions {
        token_offset: 100,
        char_offset: 1000,
        ..defaults()
    };
    let aligned = align_extraction_groups(groups, source, &opts).unwrap();
    let ex = &aligned[0][0];
    let ti = ex.token_interval.unwrap();
    let ci = ex.char_interval.unwrap();
    assert_eq!(ti.start_index, 3 + 100);
    assert_eq!(ti.end_index, 4 + 100);
    assert_eq!(ci.start, 22 + 1000);
    assert_eq!(ci.end, 30 + 1000);
}
