//! DateTime dtype support

/// Time unit for datetime
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TimeUnit {
    /// Year
    Year,
    /// Month
    Month,
    /// Week
    Week,
    /// Day
    Day,
    /// Hour
    Hour,
    /// Minute
    Minute,
    /// Second
    Second,
    /// Millisecond
    Millisecond,
    /// Microsecond
    Microsecond,
    /// Nanosecond
    Nanosecond,
}

impl TimeUnit {
    /// Get nanoseconds per unit
    pub fn nanoseconds_per_unit(&self) -> i64 {
        match self {
            TimeUnit::Year => 365 * 24 * 3600 * 1_000_000_000,
            TimeUnit::Month => 30 * 24 * 3600 * 1_000_000_000,
            TimeUnit::Week => 7 * 24 * 3600 * 1_000_000_000,
            TimeUnit::Day => 24 * 3600 * 1_000_000_000,
            TimeUnit::Hour => 3600 * 1_000_000_000,
            TimeUnit::Minute => 60 * 1_000_000_000,
            TimeUnit::Second => 1_000_000_000,
            TimeUnit::Millisecond => 1_000_000,
            TimeUnit::Microsecond => 1_000,
            TimeUnit::Nanosecond => 1,
        }
    }
}

