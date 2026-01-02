//! DateTime arithmetic operations

/// Add timedelta to datetime
///
/// Returns new datetime value
pub fn datetime_add_timedelta(dt: i64, td: i64) -> i64 {
    dt + td
}

/// Subtract datetime from datetime
///
/// Returns timedelta in nanoseconds
pub fn datetime_subtract_datetime(dt1: i64, dt2: i64) -> i64 {
    dt1 - dt2
}

/// Subtract timedelta from datetime
///
/// Returns new datetime value
pub fn datetime_subtract_timedelta(dt: i64, td: i64) -> i64 {
    dt - td
}

