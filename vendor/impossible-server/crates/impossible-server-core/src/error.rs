use std::{error::Error, fmt};

/// Stable, workload-neutral failure categories suitable for public protocols.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum ErrorCode {
    /// The request is structurally or semantically invalid.
    InvalidRequest,
    /// A required workload or dependency is not ready.
    Unavailable,
    /// A bounded queue or concurrency limit has been reached.
    Overloaded,
    /// The caller or server cancelled the operation.
    Cancelled,
    /// The operation exceeded its deadline.
    DeadlineExceeded,
    /// An unexpected internal failure occurred.
    Internal,
}

impl ErrorCode {
    /// Returns the stable lower-snake-case representation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::InvalidRequest => "invalid_request",
            Self::Unavailable => "unavailable",
            Self::Overloaded => "overloaded",
            Self::Cancelled => "cancelled",
            Self::DeadlineExceeded => "deadline_exceeded",
            Self::Internal => "internal",
        }
    }
}

/// Stable guidance describing whether a retry can be useful.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Retryability {
    /// Retrying the same request cannot succeed.
    Never,
    /// Retrying later may succeed.
    Retryable,
    /// No safe retry claim can be made.
    Unknown,
}

/// Privacy-reviewed error data safe to return to a client or normal log.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PublicError {
    /// Stable machine-readable code.
    pub code: ErrorCode,
    /// Stable retry guidance.
    pub retryability: Retryability,
    /// Static message containing no request or runtime detail.
    pub message: &'static str,
}

impl PublicError {
    /// Returns the canonical public representation for a stable code.
    #[must_use]
    pub const fn for_code(code: ErrorCode) -> Self {
        match code {
            ErrorCode::InvalidRequest => {
                Self::new(code, Retryability::Never, "the request is invalid")
            }
            ErrorCode::Unavailable => Self::new(
                code,
                Retryability::Retryable,
                "the requested service is unavailable",
            ),
            ErrorCode::Overloaded => Self::new(
                code,
                Retryability::Retryable,
                "the service is temporarily overloaded",
            ),
            ErrorCode::Cancelled => {
                Self::new(code, Retryability::Never, "the request was cancelled")
            }
            ErrorCode::DeadlineExceeded => Self::new(
                code,
                Retryability::Never,
                "the request deadline was exceeded",
            ),
            ErrorCode::Internal => Self::new(
                code,
                Retryability::Unknown,
                "an internal server error occurred",
            ),
        }
    }

    const fn new(code: ErrorCode, retryability: Retryability, message: &'static str) -> Self {
        Self {
            code,
            retryability,
            message,
        }
    }
}

/// A public failure with a private diagnostic source.
pub struct DiagnosticError {
    public: PublicError,
    source: Option<Box<dyn Error + Send + Sync + 'static>>,
}

impl DiagnosticError {
    /// Creates an error without a private diagnostic.
    #[must_use]
    pub const fn public(code: ErrorCode) -> Self {
        Self {
            public: PublicError::for_code(code),
            source: None,
        }
    }

    /// Creates an error carrying a diagnostic that is never formatted by `Debug` or `Display`.
    #[must_use]
    pub fn with_source(code: ErrorCode, source: impl Error + Send + Sync + 'static) -> Self {
        Self {
            public: PublicError::for_code(code),
            source: Some(Box::new(source)),
        }
    }

    /// Returns the only representation transports should serialize.
    #[must_use]
    pub const fn public_error(&self) -> &PublicError {
        &self.public
    }

    /// Returns private diagnostics for explicitly access-controlled telemetry.
    #[must_use]
    pub fn diagnostic_source(&self) -> Option<&(dyn Error + Send + Sync + 'static)> {
        self.source.as_deref()
    }
}

impl fmt::Debug for DiagnosticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DiagnosticError")
            .field("public", &self.public)
            .field("source", &self.source.as_ref().map(|_| "[REDACTED]"))
            .finish()
    }
}

impl fmt::Display for DiagnosticError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(self.public.message)
    }
}

// Intentionally do not expose the diagnostic through `Error::source`. Generic error reporters
// commonly traverse that chain into ordinary logs and client responses. Callers authorized to
// handle private diagnostics must opt in through `diagnostic_source` instead.
impl Error for DiagnosticError {}

#[cfg(test)]
mod tests {
    use super::{DiagnosticError, ErrorCode};

    #[test]
    fn ordinary_formatting_redacts_private_diagnostics() {
        let error = DiagnosticError::with_source(
            ErrorCode::Internal,
            std::io::Error::other("private-diagnostic-sentinel"),
        );
        assert!(!format!("{error:?}").contains("private-diagnostic-sentinel"));
        assert!(!error.to_string().contains("private-diagnostic-sentinel"));
        assert!(error.diagnostic_source().is_some());
        assert!(std::error::Error::source(&error).is_none());
    }

    #[test]
    fn public_codes_are_stable() {
        assert_eq!(ErrorCode::InvalidRequest.as_str(), "invalid_request");
        assert_eq!(ErrorCode::Unavailable.as_str(), "unavailable");
        assert_eq!(ErrorCode::Overloaded.as_str(), "overloaded");
        assert_eq!(ErrorCode::Cancelled.as_str(), "cancelled");
        assert_eq!(ErrorCode::DeadlineExceeded.as_str(), "deadline_exceeded");
        assert_eq!(ErrorCode::Internal.as_str(), "internal");
    }
}
