//! Content-addressed cache for per-chunk LLM responses.
//!
//! At scale, most chunks are boilerplate that repeats across
//! documents — IRS tax forms share "See Pub. 17", standard
//! definitions, and disclaimers in essentially every document.
//! Caching LLM responses keyed on `(prompt_description_hash +
//! chunk_text_hash)` eliminates redundant calls and turns a
//! 1000-document batch's LLM cost into roughly one document's
//! worth of unique content plus a thin diff.
//!
//! This module defines the [`ChunkCache`] trait plus an in-memory
//! implementation. Consumers who want persistent or shared caches
//! (`SQLite`, Redis, S3) implement the trait themselves.
//!
//! # Cache key
//!
//! The key is a SHA-256 hex digest of:
//!
//! ```text
//! sha256(
//!     prompt_description
//!     || "\n\x1E\n"                  // record separator
//!     || schema_fingerprint
//!     || "\n\x1E\n"
//!     || chunk_text
//! )
//! ```
//!
//! - `prompt_description` — the user-authored instructions. Change
//!   it and the cache is automatically invalidated.
//! - `schema_fingerprint` — a short string describing the format
//!   handler configuration (format type, wrapper key, fence mode,
//!   attribute suffix). Change the output format and the cache
//!   invalidates.
//! - `chunk_text` — the chunk itself.
//!
//! The effect: as long as the prompt and schema are stable, two
//! chunks with identical text reuse the same LLM response.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::Mutex;

use sha2::{Digest, Sha256};

/// Cache key — a 64-character hex SHA-256 digest.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey(pub String);

impl CacheKey {
    /// Derive a cache key from the three inputs.
    #[must_use]
    pub fn from(description: &str, schema_fingerprint: &str, chunk_text: &str) -> Self {
        let mut hasher = Sha256::new();
        hasher.update(description.as_bytes());
        hasher.update(b"\n\x1E\n");
        hasher.update(schema_fingerprint.as_bytes());
        hasher.update(b"\n\x1E\n");
        hasher.update(chunk_text.as_bytes());
        let digest = hasher.finalize();
        let mut hex = String::with_capacity(digest.len() * 2);
        for b in digest {
            let _ = write!(hex, "{b:02x}");
        }
        Self(hex)
    }

    /// Borrow the hex string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Per-chunk LLM response cache. Implementations should be cheap to
/// clone so the pipeline can hand shared references to each chunk's
/// task.
pub trait ChunkCache: Send + Sync + std::fmt::Debug {
    /// Look up a previously-cached response for the key. Returns
    /// `None` on a miss.
    fn get(&self, key: &CacheKey) -> Option<String>;

    /// Store a response under the key. Overwrites any previous
    /// value for the same key.
    fn put(&self, key: &CacheKey, response: String);

    /// Approximate entry count. Used by the pipeline's report
    /// layer for observability. Implementations that can't cheaply
    /// report length may return `None`.
    fn len(&self) -> Option<usize> {
        None
    }

    /// Convenience: whether the cache is empty. Default
    /// implementation consults [`Self::len`] and treats `None` as
    /// "unknown, assume not empty".
    fn is_empty(&self) -> bool {
        self.len() == Some(0)
    }
}

/// Simple in-memory `ChunkCache` backed by a `HashMap<CacheKey, String>`
/// behind a `Mutex`. Lives for the duration of the pipeline run
/// unless the caller holds the cache outside and reuses it across
/// runs.
#[derive(Debug, Default)]
pub struct InMemoryChunkCache {
    entries: Mutex<HashMap<CacheKey, String>>,
}

impl InMemoryChunkCache {
    /// Construct an empty cache.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Drop every entry.
    ///
    /// # Panics
    ///
    /// Panics if the internal mutex is poisoned, which would only
    /// happen if a previous holder panicked while holding the lock.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }
}

impl ChunkCache for InMemoryChunkCache {
    fn get(&self, key: &CacheKey) -> Option<String> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    fn put(&self, key: &CacheKey, response: String) {
        self.entries.lock().unwrap().insert(key.clone(), response);
    }

    fn len(&self) -> Option<usize> {
        Some(self.entries.lock().unwrap().len())
    }
}

/// No-op cache that never stores anything. Default when the caller
/// doesn't configure a cache.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpChunkCache;

impl ChunkCache for NoOpChunkCache {
    fn get(&self, _key: &CacheKey) -> Option<String> {
        None
    }
    fn put(&self, _key: &CacheKey, _response: String) {}
    fn len(&self) -> Option<usize> {
        Some(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn cache_key_is_deterministic() {
        let a = CacheKey::from("desc", "schema", "chunk");
        let b = CacheKey::from("desc", "schema", "chunk");
        assert_eq!(a, b);
        assert_eq!(a.as_str().len(), 64);
    }

    #[test]
    fn cache_key_distinguishes_inputs() {
        let a = CacheKey::from("desc1", "schema", "chunk");
        let b = CacheKey::from("desc2", "schema", "chunk");
        let c = CacheKey::from("desc1", "schema2", "chunk");
        let d = CacheKey::from("desc1", "schema", "chunk2");
        assert_ne!(a, b);
        assert_ne!(a, c);
        assert_ne!(a, d);
    }

    #[test]
    fn in_memory_cache_round_trip() {
        let cache = InMemoryChunkCache::new();
        let key = CacheKey::from("d", "s", "c");
        assert_eq!(cache.get(&key), None);
        cache.put(&key, "response".to_owned());
        assert_eq!(cache.get(&key).as_deref(), Some("response"));
        assert_eq!(cache.len(), Some(1));
    }

    #[test]
    fn noop_cache_never_stores() {
        let cache = NoOpChunkCache;
        let key = CacheKey::from("d", "s", "c");
        cache.put(&key, "x".to_owned());
        assert_eq!(cache.get(&key), None);
    }
}
