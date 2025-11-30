class PynesError(Exception):
    """Base exception for all PyNES related errors."""
    pass

class ExitException(PynesError):
    """Exception to signal the application to exit."""
    pass