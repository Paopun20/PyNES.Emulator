import ctypes
import threading
from typing import Type
from returns.result import Result, Success, Failure


def thread_exception(thread: threading.Thread, exctype: Type[BaseException]) -> Result[bool, Exception]:
    """Raises an exception in the context of the given thread, returning Result."""
    if not thread.is_alive():
        return Failure(RuntimeError("Thread is not alive"))

    tid = thread.ident
    if tid is None:
        return Failure(RuntimeError("Thread has no valid ident"))

    try:
        res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    except Exception as e:
        return Failure(e)

    if res == 0:
        return Failure(ValueError("Thread ID not found"))
    elif res > 1:
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        return Failure(SystemError("Failed to raise exception in thread"))

    return Success(True)
