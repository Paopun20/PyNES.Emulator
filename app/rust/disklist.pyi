from typing import List, TypedDict


class Disk(TypedDict):
    name: str
    path: str
    mapper: int


class DiskList(TypedDict):
    disks: List[Disk]
