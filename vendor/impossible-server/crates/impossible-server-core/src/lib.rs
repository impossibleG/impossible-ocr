//! Modality-neutral primitives for bounded, privacy-safe local services.

mod cancellation;
mod error;
mod health;
mod limits;
mod request;
mod shutdown;

pub use cancellation::CancellationToken;
pub use error::{DiagnosticError, ErrorCode, PublicError, Retryability};
pub use health::{HealthRegistry, HealthSnapshot, ProcessState, ReadinessReason};
pub use limits::{LimitError, ServerLimits};
pub use request::{
    DeadlineError, RequestContext, RequestId, RequestIdExhausted, RequestIdSource, RequestStop,
};
pub use shutdown::{DrainOutcome, ShutdownGate, WorkGuard};
