import ctypes
import threading

def make_thread_exception(thread: threading.Thread, exctype: ctypes.py_object):
    """Raises an exception in the context of the given thread."""
    if not thread.is_alive():
        return
    tid = thread.ident
    res = ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), ctypes.py_object(exctype))
    if res == 0:
        raise ValueError("Thread ID not found")
    elif res > 1:
        # Revert in case of error
        ctypes.pythonapi.PyThreadState_SetAsyncExc(ctypes.c_long(tid), None)
        raise SystemError("Failed to raise exception in thread")
    return True