__version__: tuple[int, int, int, str, int] = (0, 0, 0, "dev", 0)
__dev__: bool = __version__[3] != "stable"
__version_string__: str = (
    f"{__version__[0]}.{__version__[1]}.{__version__[2]}"
    + (f"-{__version__[3]}.{__version__[4]}" if __dev__ else "")
)