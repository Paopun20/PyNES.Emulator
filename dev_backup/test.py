from disklist import DiskList
from pathlib import Path

dlist = DiskList("trace.log")
dlist.clean()

import datetime

dlist.append(f"[{datetime.datetime.now()}] Hello World")

print(dlist[0])