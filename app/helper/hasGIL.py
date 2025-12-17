import sys
from typing import Literal


def hasGIL() -> Literal[-1, 0, 1]:
    """
    Check whether the current Python build uses the Global Interpreter Lock (GIL).

    Returns:
        int:
            1 - GIL is enabled
            0 - GIL is disabled (no-GIL build)
           -1 - Unable to determine (likely unsupported Python version)
    """
    if hasattr(sys, "_is_gil_enabled"):
        flag = sys._is_gil_enabled()
    else:
        flag = None

    match flag:
        case True:
            return 1
        case False:
            return 0
        case None:
            return -1