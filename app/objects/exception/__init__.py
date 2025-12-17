class PynesError(BaseException):
    """Base exception for all PyNES related errors."""
    pass

class ExitException(BaseException):
    """Exception to signal the application to exit."""
    pass