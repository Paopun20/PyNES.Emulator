import sysconfig
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
    flag = sysconfig.get_config_vars().get('Py_GIL_DISABLED')
    if flag is None:
        return -1             # older Python / cannot determine

    if flag == 1:
        return 0              # build has GIL disabled → no-GIL build
    else:
        return 1              # build uses GIL by default
