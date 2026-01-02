//! Statistical operations
//!
//! This module provides statistical functions like percentile, median, std, etc.

mod percentile;
mod central;
mod dispersion;
mod correlation;
mod histogram;

use crate::array::ArrayError;

pub use crate::array::ArrayError as StatisticsArrayError;

/// Statistics error
#[derive(Debug, Clone)]
pub enum StatisticsError {
    /// Array error
    ArrayError(ArrayError),
    /// Invalid percentile value (must be 0-100)
    InvalidPercentile,
    /// Unsupported type
    UnsupportedType,
}

impl std::fmt::Display for StatisticsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            StatisticsError::ArrayError(e) => write!(f, "Array error: {}", e),
            StatisticsError::InvalidPercentile => write!(f, "Invalid percentile (must be 0-100)"),
            StatisticsError::UnsupportedType => write!(f, "Unsupported type"),
        }
    }
}

impl std::error::Error for StatisticsError {}

impl From<ArrayError> for StatisticsError {
    fn from(err: ArrayError) -> Self {
        StatisticsError::ArrayError(err)
    }
}

pub use percentile::*;
pub use central::*;
pub use dispersion::*;
pub use correlation::*;
pub use histogram::*;

