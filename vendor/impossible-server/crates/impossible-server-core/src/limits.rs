use std::{error::Error, fmt, num::NonZeroUsize, time::Duration};

const MAX_TIMEOUT: Duration = Duration::from_secs(24 * 60 * 60);

/// Sanitized configuration validation failure.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LimitError {
    field: &'static str,
}

impl LimitError {
    /// Returns the invalid field without echoing its value.
    #[must_use]
    pub const fn field(self) -> &'static str {
        self.field
    }
}

impl fmt::Display for LimitError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "{} is outside the supported bound", self.field)
    }
}

impl Error for LimitError {}

/// Validated server resource and lifecycle bounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ServerLimits {
    max_request_bytes: NonZeroUsize,
    queue_capacity: NonZeroUsize,
    max_concurrent_requests: NonZeroUsize,
    request_timeout: Duration,
    shutdown_timeout: Duration,
}

impl ServerLimits {
    /// Largest encoded request accepted by the generic boundary (one GiB).
    pub const MAX_REQUEST_BYTES: usize = 1024 * 1024 * 1024;
    /// Largest queue supported by the generic admission primitives.
    pub const MAX_QUEUE_CAPACITY: usize = 1_000_000;
    /// Largest concurrency supported by the generic admission primitives.
    ///
    /// This is deliberately below [`tokio::sync::Semaphore::MAX_PERMITS`].
    pub const MAX_CONCURRENT_REQUESTS: usize = 1_000_000;

    /// Builds a fully bounded limit set.
    ///
    /// # Errors
    /// Returns a sanitized error for zero values, sizes above the associated safe maxima, or
    /// timeouts beyond one day.
    pub fn new(
        max_request_bytes: usize,
        queue_capacity: usize,
        max_concurrent_requests: usize,
        request_timeout: Duration,
        shutdown_timeout: Duration,
    ) -> Result<Self, LimitError> {
        Ok(Self {
            max_request_bytes: bounded_size(
                max_request_bytes,
                Self::MAX_REQUEST_BYTES,
                "max_request_bytes",
            )?,
            queue_capacity: bounded_size(
                queue_capacity,
                Self::MAX_QUEUE_CAPACITY,
                "queue_capacity",
            )?,
            max_concurrent_requests: bounded_size(
                max_concurrent_requests,
                Self::MAX_CONCURRENT_REQUESTS,
                "max_concurrent_requests",
            )?,
            request_timeout: bounded_duration(request_timeout, "request_timeout")?,
            shutdown_timeout: bounded_duration(shutdown_timeout, "shutdown_timeout")?,
        })
    }

    /// Maximum encoded request size.
    #[must_use]
    pub const fn max_request_bytes(self) -> usize {
        self.max_request_bytes.get()
    }

    /// Maximum queued operations.
    #[must_use]
    pub const fn queue_capacity(self) -> usize {
        self.queue_capacity.get()
    }

    /// Maximum concurrently executing operations.
    #[must_use]
    pub const fn max_concurrent_requests(self) -> usize {
        self.max_concurrent_requests.get()
    }

    /// Maximum request lifetime.
    #[must_use]
    pub const fn request_timeout(self) -> Duration {
        self.request_timeout
    }

    /// Maximum graceful shutdown duration.
    #[must_use]
    pub const fn shutdown_timeout(self) -> Duration {
        self.shutdown_timeout
    }
}

impl Default for ServerLimits {
    fn default() -> Self {
        Self::new(
            1024 * 1024,
            128,
            8,
            Duration::from_secs(30),
            Duration::from_secs(10),
        )
        .unwrap_or_else(|_| unreachable!("static defaults are valid"))
    }
}

fn non_zero(value: usize, field: &'static str) -> Result<NonZeroUsize, LimitError> {
    NonZeroUsize::new(value).ok_or(LimitError { field })
}

fn bounded_size(
    value: usize,
    maximum: usize,
    field: &'static str,
) -> Result<NonZeroUsize, LimitError> {
    let value = non_zero(value, field)?;
    if value.get() > maximum {
        return Err(LimitError { field });
    }
    Ok(value)
}

fn bounded_duration(value: Duration, field: &'static str) -> Result<Duration, LimitError> {
    if value.is_zero() || value > MAX_TIMEOUT {
        return Err(LimitError { field });
    }
    Ok(value)
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::ServerLimits;

    #[test]
    fn every_bound_rejects_zero() {
        assert!(
            ServerLimits::new(0, 1, 1, Duration::from_secs(1), Duration::from_secs(1)).is_err()
        );
        assert!(
            ServerLimits::new(1, 0, 1, Duration::from_secs(1), Duration::from_secs(1)).is_err()
        );
        assert!(
            ServerLimits::new(1, 1, 0, Duration::from_secs(1), Duration::from_secs(1)).is_err()
        );
        assert!(ServerLimits::new(1, 1, 1, Duration::ZERO, Duration::from_secs(1)).is_err());
        assert!(ServerLimits::new(1, 1, 1, Duration::from_secs(1), Duration::ZERO).is_err());
    }

    #[test]
    fn size_bounds_reject_values_above_safe_maxima() {
        let timeout = Duration::from_secs(1);
        assert!(
            ServerLimits::new(
                ServerLimits::MAX_REQUEST_BYTES.saturating_add(1),
                1,
                1,
                timeout,
                timeout,
            )
            .is_err()
        );
        assert!(
            ServerLimits::new(
                1,
                ServerLimits::MAX_QUEUE_CAPACITY.saturating_add(1),
                1,
                timeout,
                timeout,
            )
            .is_err()
        );
        assert!(
            ServerLimits::new(
                1,
                1,
                ServerLimits::MAX_CONCURRENT_REQUESTS.saturating_add(1),
                timeout,
                timeout,
            )
            .is_err()
        );
        let limits = ServerLimits::new(
            ServerLimits::MAX_REQUEST_BYTES,
            ServerLimits::MAX_QUEUE_CAPACITY,
            ServerLimits::MAX_CONCURRENT_REQUESTS,
            timeout,
            timeout,
        );
        assert!(limits.is_ok());
        if let Ok(limits) = limits {
            let semaphore = tokio::sync::Semaphore::new(limits.max_concurrent_requests());
            assert_eq!(
                semaphore.available_permits(),
                limits.max_concurrent_requests()
            );
        }
    }
}
