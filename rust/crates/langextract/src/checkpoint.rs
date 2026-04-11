//! Document-level checkpointing for resumable batch runs.
//!
//! When you're running thousands of documents through the pipeline,
//! a mid-batch crash or ctrl-C must not force a restart from scratch.
//! The [`Checkpoint`] trait defines the minimum surface the batch
//! runner needs:
//!
//! - `is_completed(id)` — has this document already been processed?
//! - `mark_completed(id)` — record a successful document.
//! - `completed_ids()` — enumerate everything seen so far (used at
//!   startup to prune the input queue).
//!
//! Two implementations ship here:
//!
//! - [`InMemoryCheckpoint`] — holds the set in a `Mutex<HashSet>`.
//!   Useful for tests and for short-running batches where a crash
//!   just means a full retry.
//! - [`JsonlCheckpoint`] — appends one line per completed document
//!   to a JSONL file, fsync-on-append. Cheap, crash-safe, and trivial
//!   to inspect by hand. Good enough for any batch that fits on a
//!   single machine.
//!
//! For distributed runs, implement the trait over Redis, a relational
//! database, or S3 object existence.

use std::collections::HashSet;
use std::fs::OpenOptions;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use crate::error::ExtractError;

/// Opaque document identifier used by the checkpoint layer.
///
/// This is just a string — callers are free to use a document hash,
/// a filesystem path, a database primary key, or any other stable
/// identifier, as long as it's unique per document within the batch.
pub type CheckpointId = String;

/// Trait implemented by checkpoint backends.
pub trait Checkpoint: Send + Sync + std::fmt::Debug {
    /// Has this document already been fully processed?
    fn is_completed(&self, id: &str) -> bool;

    /// Record a document as fully processed. Must be idempotent —
    /// calling `mark_completed` twice for the same id is a no-op.
    ///
    /// # Errors
    ///
    /// Returns an error from the underlying storage backend; wrapped
    /// in [`ExtractError`] so pipeline callers can `?`-chain.
    fn mark_completed(&self, id: &str) -> Result<(), ExtractError>;

    /// Enumerate every id previously recorded as completed. The
    /// pipeline calls this once at startup to filter its input
    /// queue.
    fn completed_ids(&self) -> Vec<CheckpointId>;
}

/// In-memory checkpoint — fast, non-persistent.
#[derive(Debug, Default)]
pub struct InMemoryCheckpoint {
    inner: Mutex<HashSet<CheckpointId>>,
}

impl InMemoryCheckpoint {
    /// Construct an empty checkpoint.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

impl Checkpoint for InMemoryCheckpoint {
    fn is_completed(&self, id: &str) -> bool {
        self.inner.lock().unwrap().contains(id)
    }

    fn mark_completed(&self, id: &str) -> Result<(), ExtractError> {
        self.inner.lock().unwrap().insert(id.to_owned());
        Ok(())
    }

    fn completed_ids(&self) -> Vec<CheckpointId> {
        self.inner.lock().unwrap().iter().cloned().collect()
    }
}

/// No-op checkpoint — always reports "not completed". Default when
/// the caller doesn't configure one.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpCheckpoint;

impl Checkpoint for NoOpCheckpoint {
    fn is_completed(&self, _id: &str) -> bool {
        false
    }
    fn mark_completed(&self, _id: &str) -> Result<(), ExtractError> {
        Ok(())
    }
    fn completed_ids(&self) -> Vec<CheckpointId> {
        Vec::new()
    }
}

/// JSONL-backed checkpoint. Appends one line per completed document,
/// fsync-on-append, crash-safe. Format:
///
/// ```text
/// {"id":"doc_abc123"}
/// {"id":"doc_def456"}
/// ```
#[derive(Debug)]
pub struct JsonlCheckpoint {
    path: PathBuf,
    in_memory: Mutex<HashSet<CheckpointId>>,
}

impl JsonlCheckpoint {
    /// Open (or create) a JSONL checkpoint file. Reads any existing
    /// entries into memory so subsequent `is_completed` calls are
    /// O(1).
    ///
    /// # Errors
    ///
    /// Returns an error if the file can't be opened or any existing
    /// line can't be parsed as the expected `{"id": "..."}` shape.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, ExtractError> {
        let path = path.as_ref().to_path_buf();
        let mut in_memory = HashSet::new();
        if path.exists() {
            let file = OpenOptions::new()
                .read(true)
                .open(&path)
                .map_err(|e| {
                    ExtractError::Checkpoint(format!("open {}: {e}", path.display()))
                })?;
            let reader = BufReader::new(file);
            for (i, line) in reader.lines().enumerate() {
                let line = line
                    .map_err(|e| ExtractError::Checkpoint(format!("read line {i}: {e}")))?;
                if line.trim().is_empty() {
                    continue;
                }
                let parsed: serde_json::Value = serde_json::from_str(&line)
                    .map_err(|e| ExtractError::Checkpoint(format!("parse line {i}: {e}")))?;
                if let Some(id) = parsed.get("id").and_then(|v| v.as_str()) {
                    in_memory.insert(id.to_owned());
                }
            }
        }
        Ok(Self {
            path,
            in_memory: Mutex::new(in_memory),
        })
    }
}

impl Checkpoint for JsonlCheckpoint {
    fn is_completed(&self, id: &str) -> bool {
        self.in_memory.lock().unwrap().contains(id)
    }

    fn mark_completed(&self, id: &str) -> Result<(), ExtractError> {
        {
            let mut guard = self.in_memory.lock().unwrap();
            if !guard.insert(id.to_owned()) {
                return Ok(()); // already present; idempotent
            }
        }
        let mut file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&self.path)
            .map_err(|e| {
                ExtractError::Checkpoint(format!("append {}: {e}", self.path.display()))
            })?;
        let line = serde_json::json!({ "id": id });
        writeln!(file, "{line}")
            .map_err(|e| ExtractError::Checkpoint(format!("write: {e}")))?;
        file.sync_all()
            .map_err(|e| ExtractError::Checkpoint(format!("fsync: {e}")))?;
        Ok(())
    }

    fn completed_ids(&self) -> Vec<CheckpointId> {
        self.in_memory.lock().unwrap().iter().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use pretty_assertions::assert_eq;

    #[test]
    fn in_memory_round_trip() {
        let cp = InMemoryCheckpoint::new();
        assert!(!cp.is_completed("a"));
        cp.mark_completed("a").unwrap();
        assert!(cp.is_completed("a"));
        cp.mark_completed("a").unwrap(); // idempotent
        assert_eq!(cp.completed_ids().len(), 1);
    }

    #[test]
    fn jsonl_persists_across_reopens() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("ckpt.jsonl");

        let cp1 = JsonlCheckpoint::open(&path).unwrap();
        cp1.mark_completed("doc_1").unwrap();
        cp1.mark_completed("doc_2").unwrap();
        assert!(cp1.is_completed("doc_1"));

        let cp2 = JsonlCheckpoint::open(&path).unwrap();
        assert!(cp2.is_completed("doc_1"));
        assert!(cp2.is_completed("doc_2"));
        assert!(!cp2.is_completed("doc_3"));
        assert_eq!(cp2.completed_ids().len(), 2);
    }

    #[test]
    fn noop_never_completes() {
        let cp = NoOpCheckpoint;
        cp.mark_completed("a").unwrap();
        assert!(!cp.is_completed("a"));
        assert!(cp.completed_ids().is_empty());
    }
}
