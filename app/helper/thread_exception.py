import ctypes
import threading
from typing import Optional
from returns.result import Result, Success, Failure

def thread_exception(thread: threading.Thread, exception: BaseException) -> Result[bool, Exception]:
    """
    Raise an exception in the target thread.
    
    Args:
        thread: Thread to interrupt
        exception: Exception instance to raise (must be BaseException subclass instance)
    
    Returns:
        Success(True) on probable success, Failure with error otherwise.
    """
    if not isinstance(exception, BaseException):
        return Failure(TypeError("exception must be an instance of BaseException"))
    
    if not thread.is_alive():
        return Failure(RuntimeError("Thread is not alive"))
    
    tid: Optional[int] = thread.ident
    if tid is None:
        return Failure(RuntimeError("Thread has no valid ident"))
    
    try:
        # Use c_ulong for thread ID (matches C API's unsigned long)
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(
            ctypes.c_ulong(tid),
            ctypes.py_object(exception)
        )
    except Exception as e:
        return Failure(e)
    
    if res == 0:
        return Failure(ValueError(f"Thread ID {tid} not found"))
    elif res > 1:
        # Clear the exception if it was set on multiple threads (error case)
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_ulong(tid), None)
        return Failure(SystemError("Critical failure: Exception propagated to multiple threads"))
    
    return Success(True)