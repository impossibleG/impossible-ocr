use std::{
    collections::BTreeMap,
    sync::{Arc, RwLock},
};

/// Process lifecycle state exposed through health endpoints.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProcessState {
    /// Process initialization has not completed.
    Starting,
    /// Process is accepting work.
    Running,
    /// Process is draining and accepts no new work.
    Draining,
    /// Process shutdown is complete.
    Stopped,
}

/// Stable aggregate readiness reason.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReadinessReason {
    /// The process has not entered the running state.
    ProcessNotRunning,
    /// At least one registered workload component is unavailable.
    ComponentUnavailable,
}

impl ReadinessReason {
    /// Returns a stable public code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::ProcessNotRunning => "process_not_running",
            Self::ComponentUnavailable => "component_unavailable",
        }
    }
}

#[derive(Debug)]
struct State {
    process: ProcessState,
    components: BTreeMap<String, bool>,
}

impl Default for State {
    fn default() -> Self {
        Self {
            process: ProcessState::Starting,
            components: BTreeMap::new(),
        }
    }
}

/// Aggregate health snapshot containing no private component detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HealthSnapshot {
    /// Whether the process event loop is alive.
    pub live: bool,
    /// Whether the process and every registered component can serve work.
    pub ready: bool,
    /// Stable aggregate reason when not ready.
    pub reason: Option<ReadinessReason>,
}

/// Thread-safe health registry with aggregate-only public snapshots.
#[derive(Debug, Clone, Default)]
pub struct HealthRegistry(Arc<RwLock<State>>);

impl HealthRegistry {
    /// Creates a registry in `Starting` state.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the process lifecycle state.
    pub fn set_process(&self, process: ProcessState) {
        self.0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .process = process;
    }

    /// Registers a named component as initially unavailable.
    pub fn register_component(&self, name: impl Into<String>) {
        self.0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .components
            .entry(name.into())
            .or_insert(false);
    }

    /// Updates a component, returning `false` when it was never registered.
    #[must_use]
    pub fn set_component_ready(&self, name: &str, ready: bool) -> bool {
        let mut state = self
            .0
            .write()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let Some(component) = state.components.get_mut(name) else {
            return false;
        };
        *component = ready;
        true
    }

    /// Returns a privacy-safe aggregate snapshot.
    #[must_use]
    pub fn snapshot(&self) -> HealthSnapshot {
        let state = self
            .0
            .read()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let live = state.process != ProcessState::Stopped;
        let process_ready = state.process == ProcessState::Running;
        let components_ready = state.components.values().all(|ready| *ready);
        let ready = process_ready && components_ready;
        let reason = if ready {
            None
        } else if !process_ready {
            Some(ReadinessReason::ProcessNotRunning)
        } else {
            Some(ReadinessReason::ComponentUnavailable)
        };
        HealthSnapshot {
            live,
            ready,
            reason,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::{HealthRegistry, ProcessState, ReadinessReason};

    #[test]
    fn readiness_requires_process_and_every_component() {
        let health = HealthRegistry::new();
        health.register_component("workload");
        assert_eq!(
            health.snapshot().reason,
            Some(ReadinessReason::ProcessNotRunning)
        );
        health.set_process(ProcessState::Running);
        assert_eq!(
            health.snapshot().reason,
            Some(ReadinessReason::ComponentUnavailable)
        );
        assert!(health.set_component_ready("workload", true));
        assert!(health.snapshot().ready);
        health.set_process(ProcessState::Stopped);
        assert!(!health.snapshot().live);
    }
}
