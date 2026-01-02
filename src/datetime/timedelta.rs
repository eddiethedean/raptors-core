//! Timedelta value representation

use super::TimeUnit;

/// Timedelta value
///
/// Represents a time duration
#[derive(Debug, Clone, Copy)]
pub struct TimeDelta {
    /// Value in nanoseconds
    pub nanoseconds: i64,
    /// Unit
    pub unit: TimeUnit,
}

impl TimeDelta {
    /// Create new timedelta
    pub fn new(value: i64, unit: TimeUnit) -> Self {
        let nanoseconds = value * unit.nanoseconds_per_unit();
        TimeDelta { nanoseconds, unit }
    }
    
    /// Get value in nanoseconds
    pub fn as_nanoseconds(&self) -> i64 {
        self.nanoseconds
    }
}

