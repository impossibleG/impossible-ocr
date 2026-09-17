use std::sync::{Arc, Mutex};
use std::time::Duration;

use tokio::time::Instant;

use crate::CancellationToken;

/// Result of a bounded graceful drain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrainOutcome {
    /// Every admitted operation completed within the bound.
    Graceful,
    /// The bound elapsed or the drain owner was cancelled.
    Forced,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Phase {
    Accepting,
    Draining,
    Stopped(DrainOutcome),
}

#[derive(Debug)]
struct State {
    phase: Phase,
    active: usize,
}

impl Default for State {
    fn default() -> Self {
        Self {
            phase: Phase::Accepting,
            active: 0,
        }
    }
}

#[derive(Debug, Default)]
struct Inner {
    state: Mutex<State>,
    stopped: CancellationToken,
}

impl Drop for Inner {
    fn drop(&mut self) {
        // A separately retained stop token must not wait forever after the final gate owner drops.
        let _ = self.stopped.cancel();
    }
}

/// Admission gate and bounded shutdown coordinator.
#[derive(Debug, Clone, Default)]
pub struct ShutdownGate(Arc<Inner>);

impl ShutdownGate {
    /// Creates an accepting gate.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Admits one operation or returns `None` after draining begins.
    #[must_use]
    pub fn try_enter(&self) -> Option<WorkGuard> {
        let mut state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        if state.phase != Phase::Accepting {
            return None;
        }
        state.active = state.active.checked_add(1)?;
        Some(WorkGuard { gate: self.clone() })
    }

    /// Returns whether new work is still accepted.
    #[must_use]
    pub fn is_accepting(&self) -> bool {
        self.0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .phase
            == Phase::Accepting
    }

    /// Returns a token cancelled when draining completes, is forced, or the final owner drops.
    #[must_use]
    pub fn stop_token(&self) -> CancellationToken {
        self.0.stopped.clone()
    }

    /// Stops admission and waits for admitted operations within `timeout`.
    ///
    /// Dropping this future after it begins forces shutdown, so cancellation cannot leave the gate
    /// permanently non-accepting without publishing the stop signal.
    pub async fn drain(&self, timeout: Duration) -> DrainOutcome {
        if let Some(outcome) = self.begin_drain() {
            return outcome;
        }

        let mut cancellation_guard = DrainCancellationGuard::new(self.clone());
        let outcome = if let Some(deadline) = Instant::now().checked_add(timeout) {
            if tokio::time::timeout_at(deadline, self.0.stopped.cancelled())
                .await
                .is_ok()
            {
                self.terminal_outcome()
            } else {
                self.stop_now();
                DrainOutcome::Forced
            }
        } else {
            self.0.stopped.cancelled().await;
            self.terminal_outcome()
        };
        cancellation_guard.disarm();
        outcome
    }

    /// Immediately stops admission, makes shutdown terminal, and wakes all drain callers.
    pub fn stop_now(&self) {
        {
            let mut state = self
                .0
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            if !matches!(state.phase, Phase::Stopped(_)) {
                state.phase = Phase::Stopped(DrainOutcome::Forced);
            }
        }
        let _ = self.0.stopped.cancel();
    }

    fn begin_drain(&self) -> Option<DrainOutcome> {
        let mut completed = None;
        {
            let mut state = self
                .0
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            match state.phase {
                Phase::Stopped(outcome) => return Some(outcome),
                Phase::Accepting => state.phase = Phase::Draining,
                Phase::Draining => {}
            }
            if state.active == 0 {
                state.phase = Phase::Stopped(DrainOutcome::Graceful);
                completed = Some(DrainOutcome::Graceful);
            }
        }
        if completed.is_some() {
            let _ = self.0.stopped.cancel();
        }
        completed
    }

    fn terminal_outcome(&self) -> DrainOutcome {
        let state = self
            .0
            .state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        match state.phase {
            Phase::Stopped(outcome) => outcome,
            Phase::Accepting | Phase::Draining => DrainOutcome::Forced,
        }
    }
}

#[derive(Debug)]
struct DrainCancellationGuard(Option<ShutdownGate>);

impl DrainCancellationGuard {
    fn new(gate: ShutdownGate) -> Self {
        Self(Some(gate))
    }

    fn disarm(&mut self) {
        self.0 = None;
    }
}

impl Drop for DrainCancellationGuard {
    fn drop(&mut self) {
        if let Some(gate) = self.0.take() {
            gate.stop_now();
        }
    }
}

/// RAII permit for one admitted operation.
#[derive(Debug)]
pub struct WorkGuard {
    gate: ShutdownGate,
}

impl Drop for WorkGuard {
    fn drop(&mut self) {
        let completed = {
            let mut state = self
                .gate
                .0
                .state
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner);
            state.active = state.active.saturating_sub(1);
            if state.active == 0 && state.phase == Phase::Draining {
                state.phase = Phase::Stopped(DrainOutcome::Graceful);
                true
            } else {
                false
            }
        };
        if completed {
            let _ = self.gate.0.stopped.cancel();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::{DrainOutcome, ShutdownGate};

    #[tokio::test]
    async fn drain_rejects_new_work_and_waits_for_active_guard() {
        let gate = ShutdownGate::new();
        let guard = gate.try_enter().ok_or("admission failed");
        assert!(guard.is_ok());
        let guard = guard.ok();
        let draining = gate.clone();
        let join = tokio::spawn(async move { draining.drain(Duration::from_secs(1)).await });
        tokio::task::yield_now().await;
        assert!(!gate.is_accepting());
        drop(guard);
        assert_eq!(join.await.ok(), Some(DrainOutcome::Graceful));
        assert!(gate.stop_token().is_cancelled());
    }

    #[tokio::test]
    async fn drain_is_bounded_when_work_is_retained() {
        let gate = ShutdownGate::new();
        let _guard = gate.try_enter();
        assert_eq!(
            gate.drain(Duration::from_millis(1)).await,
            DrainOutcome::Forced
        );
    }

    #[tokio::test]
    async fn stop_now_wakes_an_in_flight_drain_as_forced() {
        let gate = ShutdownGate::new();
        let _guard = gate.try_enter();
        let draining = gate.clone();
        let join = tokio::spawn(async move { draining.drain(Duration::from_secs(60)).await });
        while gate.is_accepting() {
            tokio::task::yield_now().await;
        }
        gate.stop_now();
        let outcome = tokio::time::timeout(Duration::from_millis(100), join).await;
        assert!(matches!(outcome, Ok(Ok(DrainOutcome::Forced))));
    }

    #[tokio::test]
    async fn aborted_drain_forces_terminal_stop() {
        let gate = ShutdownGate::new();
        let _guard = gate.try_enter();
        let draining = gate.clone();
        let join = tokio::spawn(async move { draining.drain(Duration::from_secs(60)).await });
        while gate.is_accepting() {
            tokio::task::yield_now().await;
        }
        join.abort();
        let _ = join.await;
        assert!(gate.stop_token().is_cancelled());
        assert!(gate.try_enter().is_none());
        assert_eq!(
            gate.drain(Duration::from_secs(1)).await,
            DrainOutcome::Forced
        );
    }

    #[tokio::test]
    async fn dropping_final_owner_cancels_a_retained_stop_token() {
        let gate = ShutdownGate::new();
        let stopped = gate.stop_token();
        drop(gate);
        assert!(stopped.is_cancelled());
    }
}
