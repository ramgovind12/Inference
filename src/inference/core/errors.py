"""Exception hierarchy shared across InferScale components."""


class InferenceError(Exception):
    """Base class for all inference-related errors."""


class BackendError(InferenceError):
    """The inference backend failed to process a request."""


class BackendUnavailableError(BackendError):
    """The inference backend is not started or cannot accept requests."""


class WorkerNotReadyError(InferenceError):
    """The worker is not in a state where it can accept requests."""


class WorkerUnreachableError(InferenceError):
    """A worker could not be contacted (connection failure or timeout)."""


class NoAvailableWorkerError(InferenceError):
    """No healthy worker could serve the request."""
