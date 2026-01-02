//! DateTime arithmetic operations

/// Add timedelta to datetime
///
/// Returns new datetime value
/// Uses checked arithmetic to prevent overflow
pub fn datetime_add_timedelta(dt: i64, td: i64) -> i64 {
    dt.saturating_add(td)
}

/// Subtract datetime from datetime
///
/// Returns timedelta in nanoseconds
/// Uses checked arithmetic to prevent overflow
pub fn datetime_subtract_datetime(dt1: i64, dt2: i64) -> i64 {
    dt1.saturating_sub(dt2)
}

/// Subtract timedelta from datetime
///
/// Returns new datetime value
/// Uses checked arithmetic to prevent overflow
pub fn datetime_subtract_timedelta(dt: i64, td: i64) -> i64 {
    dt.saturating_sub(td)
}

