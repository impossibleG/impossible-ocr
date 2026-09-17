use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

use tokio::sync::Notify;

#[derive(Debug, Default)]
struct State {
    cancelled: AtomicBool,
    notify: Notify,
}

/// Cloneable, one-way cancellation primitive with no missed wakeups.
#[derive(Debug, Clone, Default)]
pub struct CancellationToken(Arc<State>);

impl CancellationToken {
    /// Creates a live token.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Cancels all current and future observers. Returns whether this call changed the state.
    #[must_use]
    pub fn cancel(&self) -> bool {
        if self.0.cancelled.swap(true, Ordering::AcqRel) {
            return false;
        }
        self.0.notify.notify_waiters();
        true
    }

    /// Returns whether cancellation has occurred.
    #[must_use]
    pub fn is_cancelled(&self) -> bool {
        self.0.cancelled.load(Ordering::Acquire)
    }

    /// Waits until the token is cancelled.
    pub async fn cancelled(&self) {
        loop {
            let notified = self.0.notify.notified();
            if self.is_cancelled() {
                return;
            }
            notified.await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::CancellationToken;

    #[tokio::test]
    async fn cancellation_is_one_way_and_visible_to_clones() {
        let token = CancellationToken::new();
        let clone = token.clone();
        assert!(token.cancel());
        assert!(!clone.cancel());
        clone.cancelled().await;
        assert!(clone.is_cancelled());
    }
}
