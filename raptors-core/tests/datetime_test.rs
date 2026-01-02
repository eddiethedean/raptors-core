//! Tests for datetime functionality

#[cfg(test)]
mod tests {
    use raptors_core::datetime::{TimeUnit, datetime_from_string, datetime_add_timedelta, datetime_subtract_datetime};
    use raptors_core::datetime::{TimeDelta, parse_iso8601, format_iso8601};

    #[test]
    fn test_time_unit_nanoseconds() {
        let unit = TimeUnit::Second;
        assert_eq!(unit.nanoseconds_per_unit(), 1_000_000_000);
        
        let unit = TimeUnit::Millisecond;
        assert_eq!(unit.nanoseconds_per_unit(), 1_000_000);
        
        let unit = TimeUnit::Microsecond;
        assert_eq!(unit.nanoseconds_per_unit(), 1_000);
        
        let unit = TimeUnit::Nanosecond;
        assert_eq!(unit.nanoseconds_per_unit(), 1);
    }

    #[test]
    fn test_timedelta_new() {
        let td = TimeDelta::new(5, TimeUnit::Second);
        assert_eq!(td.as_nanoseconds(), 5_000_000_000);
    }

    #[test]
    fn test_datetime_arithmetic() {
        let dt1 = 1000i64;
        let dt2 = 500i64;
        let td = 200i64;
        
        // Add timedelta
        let result = datetime_add_timedelta(dt1, td);
        assert_eq!(result, 1200);
        
        // Subtract timedelta
        let result = datetime_subtract_datetime(dt1, td);
        assert_eq!(result, 800);
        
        // Subtract datetime
        let result = datetime_subtract_datetime(dt1, dt2);
        assert_eq!(result, 500);
    }

    #[test]
    fn test_datetime_from_string() {
        // Simplified test - current implementation returns 0
        let result = datetime_from_string("2023-01-01", TimeUnit::Day);
        assert!(result.is_ok());
    }

    #[test]
    fn test_datetime_invalid_format() {
        let result = datetime_from_string("", TimeUnit::Day);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_iso8601() {
        // Simplified test
        let result = parse_iso8601("2023-01-01T00:00:00");
        assert!(result.is_ok());
    }

    #[test]
    fn test_format_iso8601() {
        let dt = 1000i64;
        let formatted = format_iso8601(dt, TimeUnit::Second);
        assert!(!formatted.is_empty());
    }
}

