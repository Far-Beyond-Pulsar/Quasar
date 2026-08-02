use thiserror::Error;

/// Errors that can arise during spatial audio processing.
#[derive(Error, Debug)]
pub enum SpatialAudioError {
    /// An error reported from a compute backend.
    #[error("backend error: {0}")]
    Backend(String),

    /// An error related to material operations.
    #[error("material error: {0}")]
    Material(String),

    /// The scene description is invalid for processing.
    #[error("invalid scene: {0}")]
    InvalidScene(String),

    /// An error from the probe grid system.
    #[error("probe grid error: {0}")]
    ProbeGrid(String),

    /// An I/O error (wraps `std::io::Error`).
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),

    /// A serialization error.
    #[error("serialization error: {0}")]
    Serialize(String),

    /// A deserialization error.
    #[error("deserialization error: {0}")]
    Deserialize(String),

    /// A general error with a message string.
    #[error("{0}")]
    General(String),
}
