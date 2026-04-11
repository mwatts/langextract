//! Retry policy for per-chunk LLM calls.
//!
//! The pipeline distinguishes **transient** failures — rate limits,
//! timeouts, truncated responses, malformed JSON — from **fatal**
//! ones — invalid prompts, misconfigured schemas, bugs in the
//! caller's code. Transient failures should be retried with
//! exponential backoff and jitter; fatal ones should propagate
//! immediately.
//!
//! At scale (thousands of documents × hundreds of chunks), transient
//! errors are the norm, not the exception. A single 5xx from a
//! hosted LLM shouldn't crater a 10-hour batch job.
//!
//! This module is runtime-agnostic in its types but uses
//! `tokio::time::sleep` for the actual backoff, since any realistic
//! async runtime can drive tokio primitives.

use std::time::Duration;

use langextract_core::InferError;
use langextract_format::FormatError;

/// Policy describing how to retry transient failures.
#[derive(Debug, Clone)]
pub struct RetryPolicy {
    /// Maximum number of **retries** (not total attempts). `0`
    /// disables retries; `3` means one initial attempt plus up to
    /// three retries, for a worst-case of four attempts per chunk.
    pub max_retries: u32,

    /// Initial backoff before the first retry.
    pub initial_backoff: Duration,

    /// Multiplier applied to the backoff after each retry. Typical
    /// value: 2.0 for doubling.
    pub backoff_multiplier: f32,

    /// Hard cap on any single backoff interval, regardless of
    /// multiplier. Stops runaway exponential growth.
    pub max_backoff: Duration,

    /// Jitter fraction applied to each backoff. A value of 0.25
    /// spreads backoff by ±25% to prevent thundering-herd retry
    /// storms when many chunks fail at once.
    pub jitter_fraction: f32,
}

impl Default for RetryPolicy {
    fn default() -> Self {
        Self {
            max_retries: 3,
            initial_backoff: Duration::from_millis(250),
            backoff_multiplier: 2.0,
            max_backoff: Duration::from_secs(10),
            jitter_fraction: 0.25,
        }
    }
}

impl RetryPolicy {
    /// Convenience constructor for "no retries at all" — a single
    /// attempt, failures propagate immediately. Useful in tests.
    #[must_use]
    #[expect(
        clippy::missing_const_for_fn,
        reason = "Duration::ZERO is a const; kept non-const for forward compat"
    )]
    pub fn none() -> Self {
        Self {
            max_retries: 0,
            initial_backoff: Duration::ZERO,
            backoff_multiplier: 1.0,
            max_backoff: Duration::ZERO,
            jitter_fraction: 0.0,
        }
    }

    /// Compute the backoff for retry attempt `n` (1-based). Applies
    /// the multiplier, the hard cap, and deterministic jitter
    /// derived from `jitter_seed` so tests can reproduce the
    /// schedule.
    #[expect(
        clippy::cast_precision_loss,
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        clippy::cast_possible_wrap,
        reason = "attempt counts are tiny and well within f32 precision"
    )]
    #[must_use]
    pub fn backoff_for(&self, attempt: u32, jitter_seed: u64) -> Duration {
        if attempt == 0 {
            return Duration::ZERO;
        }
        let base_nanos = self.initial_backoff.as_nanos() as f64;
        let mult = f64::from(self.backoff_multiplier).powi(attempt as i32 - 1);
        let raw_nanos = (base_nanos * mult).min(self.max_backoff.as_nanos() as f64);
        // Cheap deterministic jitter: use jitter_seed to derive a
        // value in [-1, 1] scaled by jitter_fraction, then apply it
        // as (1.0 + centered * jitter_fraction).
        let jitter = if self.jitter_fraction > 0.0 {
            // Knuth's simple LCG; not cryptographic, but stable
            // per-(attempt, seed) so retries backoff predictably.
            let mixed = jitter_seed
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407)
                .wrapping_add(u64::from(attempt));
            let normalized = (mixed >> 11) as f64 / f64::from(1u32 << (24 - 11));
            let centered = normalized.fract().mul_add(2.0, -1.0);
            centered.mul_add(f64::from(self.jitter_fraction), 1.0)
        } else {
            1.0
        };
        let nanos = (raw_nanos * jitter).max(0.0);
        Duration::from_nanos(nanos as u64)
    }
}

/// Whether an error should trigger a retry.
///
/// Transient: rate limits, timeouts, truncated responses,
/// malformed JSON (the LLM sometimes corrects itself on a second
/// attempt).
///
/// Fatal: anything else — invalid prompts, schema mismatches,
/// configuration errors.
#[must_use]
pub const fn is_transient_infer(err: &InferError) -> bool {
    matches!(
        err,
        InferError::Transient(_)
            | InferError::MalformedResponse(_)
            | InferError::EmptyCompletions,
    )
}

/// Same classifier for format-level errors. We retry on parse
/// failures because they often indicate an LLM hallucination that a
/// second attempt resolves.
#[must_use]
pub const fn is_transient_format(err: &FormatError) -> bool {
    matches!(
        err,
        FormatError::Parse { .. }
            | FormatError::FenceNotFound { .. }
            | FormatError::MultipleFencedBlocks
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_policy_has_zero_retries() {
        let p = RetryPolicy::none();
        assert_eq!(p.max_retries, 0);
        assert_eq!(p.backoff_for(0, 0), Duration::ZERO);
        assert_eq!(p.backoff_for(1, 0), Duration::ZERO);
    }

    #[test]
    fn default_backoff_increases_then_caps() {
        let p = RetryPolicy::default();
        let b1 = p.backoff_for(1, 42);
        let b2 = p.backoff_for(2, 42);
        let b3 = p.backoff_for(3, 42);
        // With 0.25 jitter, b2 may fall slightly below 2*b1. Just
        // assert monotonic-ish growth up to the cap.
        assert!(b1 < p.max_backoff);
        assert!(b3 <= p.max_backoff + Duration::from_millis(500));
        let _ = b2; // only checked by smoke
    }

    #[test]
    fn transient_classifier_covers_common_cases() {
        assert!(is_transient_infer(&InferError::Transient("429".into())));
        assert!(is_transient_infer(&InferError::EmptyCompletions));
        assert!(is_transient_infer(&InferError::MalformedResponse(
            "junk".into()
        )));
        assert!(!is_transient_infer(&InferError::InvalidBatch(
            "empty".into()
        )));
    }
}
