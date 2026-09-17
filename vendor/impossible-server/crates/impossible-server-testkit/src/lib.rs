//! Deterministic helpers for downstream server tests.

use std::{future::Future, time::Duration};

use tokio::{net::TcpListener, time::Instant};

/// Reserves an IPv4 loopback listener without a free-port time-of-check/time-of-use race.
///
/// # Errors
/// Returns an I/O error when the listener cannot be bound.
pub async fn reserve_loopback_listener() -> std::io::Result<TcpListener> {
    TcpListener::bind("127.0.0.1:0").await
}

/// Polls an async predicate until it succeeds or the total deadline elapses.
///
/// A pending predicate and the retry interval are both capped by the same deadline. An
/// unrepresentable timeout fails closed by returning `false`.
pub async fn eventually<F, Fut>(timeout: Duration, interval: Duration, mut predicate: F) -> bool
where
    F: FnMut() -> Fut,
    Fut: Future<Output = bool>,
{
    let Some(deadline) = Instant::now().checked_add(timeout) else {
        return false;
    };
    loop {
        match tokio::time::timeout_at(deadline, predicate()).await {
            Ok(true) => return true,
            Ok(false) => {}
            Err(_) => return false,
        }

        let now = Instant::now();
        let wake = now
            .checked_add(interval)
            .map_or(deadline, |wake| wake.min(deadline));
        tokio::time::sleep_until(wake).await;
        if wake == deadline {
            return false;
        }
    }
}

/// Asserts that ordinary formatting does not reveal a private sentinel.
///
/// # Panics
/// Panics when either rendering contains the sentinel.
pub fn assert_redacted(value: &(impl std::fmt::Debug + std::fmt::Display), sentinel: &str) {
    assert!(!format!("{value:?}").contains(sentinel));
    assert!(!value.to_string().contains(sentinel));
}

#[cfg(test)]
mod tests {
    use std::{future::pending, time::Duration};

    use super::{eventually, reserve_loopback_listener};

    #[tokio::test]
    async fn listener_is_reserved_and_predicate_is_bounded() -> Result<(), std::io::Error> {
        let listener = reserve_loopback_listener().await?;
        assert_ne!(listener.local_addr()?.port(), 0);
        let mut attempts = 0_u8;
        assert!(
            eventually(Duration::from_secs(1), Duration::from_millis(1), || {
                attempts = attempts.saturating_add(1);
                async move { attempts == 2 }
            },)
            .await
        );
        Ok(())
    }

    #[tokio::test]
    async fn pending_predicate_and_long_interval_obey_the_total_deadline() {
        let pending_result = tokio::time::timeout(
            Duration::from_millis(100),
            eventually(Duration::from_millis(5), Duration::from_millis(1), || {
                pending::<bool>()
            }),
        )
        .await;
        assert_eq!(pending_result, Ok(false));

        let long_interval_result = tokio::time::timeout(
            Duration::from_millis(100),
            eventually(
                Duration::from_millis(5),
                Duration::from_secs(60),
                || async { false },
            ),
        )
        .await;
        assert_eq!(long_interval_result, Ok(false));
    }

    #[tokio::test]
    async fn unrepresentable_timeout_fails_closed_without_panicking() {
        assert!(!eventually(Duration::MAX, Duration::ZERO, || async { true }).await);
    }
}
