"""Readers that turn an uploaded file into one or more stacks of 2D frames.

A reader is chosen by file extension and returns :class:`StackData` -- plain
arrays and metadata, no database rows -- so reading a format and storing a stack
stay separate: ``app.services.database_access.stacks`` stores whatever any reader
produces. Adding a format (a video, a TIFF z-stack) means adding a reader here and
registering its extensions in :data:`READERS`.

Files are converted at upload: frames are written as 8-bit PNGs and the original
file is not kept. Readers therefore copy only the metadata they are told to keep,
never the whole header -- vendor formats carry patient names and IDs.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

from app.exceptions import UnsupportedStackFileError
from app.services.stack_readers.base import FrameData, StackData
from app.services.stack_readers import heidelberg

#: Lower-case file extension -> reader. A reader takes the path of the uploaded
#: file and its original name and returns every stack the file holds.
READERS: dict[str, Callable[[Path, str], list[StackData]]] = {
    ".vol": heidelberg.read_vol,
    ".e2e": heidelberg.read_e2e,
}


def supported_extensions() -> list[str]:
    return sorted(READERS)


def is_stack_file(file_name: str) -> bool:
    return Path(file_name).suffix.lower() in READERS


def read_stack_file(path: Path, original_name: str) -> list[StackData]:
    """Read every stack in an uploaded file.

    :raises UnsupportedStackFileError: if the extension has no reader or the file
        holds nothing importable.
    """
    suffix = Path(original_name).suffix.lower()
    reader = READERS.get(suffix)
    if reader is None:
        raise UnsupportedStackFileError(
            f"'{original_name}' is not a supported stack file "
            f"(supported: {', '.join(supported_extensions())})."
        )
    stacks = reader(path, original_name)
    if not stacks:
        raise UnsupportedStackFileError(f"'{original_name}' contains no volume that can be imported.")
    return stacks


__all__ = ["FrameData", "StackData", "READERS", "is_stack_file", "read_stack_file",
           "supported_extensions"]
