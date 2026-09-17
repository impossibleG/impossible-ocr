use std::{
    error::Error,
    fmt,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use tokio::time::Instant;

use crate::CancellationToken;

/// Non-zero process-local request identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RequestId(u64);

impl RequestId {
    /// Returns the numeric identifier.
    #[must_use]
    pub const fn get(self) -> u64 {
        self.0
    }
}

/// The process-local request identifier space has been exhausted.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestIdExhausted;

impl fmt::Display for RequestIdExhausted {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the process request identifier space is exhausted")
    }
}

impl Error for RequestIdExhausted {}

/// Shared monotonic request-id source.
#[derive(Debug, Clone, Default)]
pub struct RequestIdSource(Arc<AtomicU64>);

impl RequestIdSource {
    /// Returns the next non-zero identifier without ever wrapping or repeating.
    ///
    /// # Errors
    /// Returns [`RequestIdExhausted`] after `u64::MAX` identifiers have been issued.
    pub fn next(&self) -> Result<RequestId, RequestIdExhausted> {
        let previous = self
            .0
            .fetch_update(Ordering::Relaxed, Ordering::Relaxed, |value| {
                value.checked_add(1)
            })
            .map_err(|_| RequestIdExhausted)?;
        Ok(RequestId(previous + 1))
    }
}

/// A relative deadline could not be represented by the Tokio clock.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DeadlineError;

impl fmt::Display for DeadlineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str("the request deadline is outside the supported bound")
    }
}

impl Error for DeadlineError {}

/// Reason a request must stop before producing a public result.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RequestStop {
    /// Caller or server cancellation won.
    Cancelled,
    /// The request deadline elapsed.
    DeadlineExceeded,
}

/// Identity and cooperative stop controls owned by one request.
#[derive(Debug, Clone)]
pub struct RequestContext {
    id: RequestId,
    cancellation: CancellationToken,
    deadline: Option<Instant>,
}

impl RequestContext {
    /// Creates a request context with an optional relative deadline.
    ///
    /// # Errors
    /// Returns [`DeadlineError`] when the relative deadline cannot be represented.
    pub fn new(
        id: RequestId,
        cancellation: CancellationToken,
        timeout: Option<Duration>,
    ) -> Result<Self, DeadlineError> {
        let now = Instant::now();
        let deadline = timeout
            .map(|duration| now.checked_add(duration).ok_or(DeadlineError))
            .transpose()?;
        Ok(Self {
            id,
            cancellation,
            deadline,
        })
    }

    /// Returns the process-local identifier.
    #[must_use]
    pub const fn id(&self) -> RequestId {
        self.id
    }

    /// Returns the request-owned cancellation token.
    #[must_use]
    pub const fn cancellation(&self) -> &CancellationToken {
        &self.cancellation
    }

    /// Returns the remaining deadline budget, if any.
    #[must_use]
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|deadline| deadline.saturating_duration_since(Instant::now()))
    }

    /// Waits for cancellation or deadline expiration with deterministic cancellation precedence.
    pub async fn stopped(&self) -> RequestStop {
        let Some(deadline) = self.deadline else {
            self.cancellation.cancelled().await;
            return RequestStop::Cancelled;
        };
        if tokio::time::timeout_at(deadline, self.cancellation.cancelled())
            .await
            .is_ok()
            || self.cancellation.is_cancelled()
        {
            RequestStop::Cancelled
        } else {
            RequestStop::DeadlineExceeded
        }
    }
}

#[cfg(test)]
mod tests {
    use std::{
        sync::{Arc, atomic::AtomicU64},
        time::Duration,
    };

    use super::{RequestContext, RequestIdSource, RequestStop};
    use crate::CancellationToken;

    #[test]
    fn ids_are_nonzero_and_shared_across_clones() -> Result<(), Box<dyn std::error::Error>> {
        let source = RequestIdSource::default();
        let clone = source.clone();
        assert_eq!(source.next()?.get(), 1);
        assert_eq!(clone.next()?.get(), 2);
        Ok(())
    }

    #[test]
    fn exhausted_ids_never_wrap_or_repeat() -> Result<(), Box<dyn std::error::Error>> {
        let source = RequestIdSource(Arc::new(AtomicU64::new(u64::MAX - 1)));
        assert_eq!(source.next()?.get(), u64::MAX);
        assert!(source.next().is_err());
        assert!(source.next().is_err());
        Ok(())
    }

    #[tokio::test]
    async fn deadline_stops_un_cancelled_request() -> Result<(), Box<dyn std::error::Error>> {
        let context = RequestContext::new(
            RequestIdSource::default().next()?,
            CancellationToken::new(),
            Some(Duration::from_millis(1)),
        )?;
        assert_eq!(context.stopped().await, RequestStop::DeadlineExceeded);
        Ok(())
    }

    #[test]
    fn unrepresentable_deadline_is_rejected_without_panicking()
    -> Result<(), Box<dyn std::error::Error>> {
        let result = RequestContext::new(
            RequestIdSource::default().next()?,
            CancellationToken::new(),
            Some(Duration::MAX),
        );
        assert!(result.is_err());
        Ok(())
    }
}
