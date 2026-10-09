"""Warnings that name the user's line of code, however deep inside tokenizerlab they start.

Reading runs as nested generators (stream, handler, parser, corpus steps), so a fixed
stacklevel lands somewhere inside the library. Python 3.12 has warnings.warn(...,
skip_file_prefixes=...) for this; tokenizerlab supports 3.11, so it walks the stack itself.
"""

import contextlib
import os
import sys
import warnings

# Frames to look past: tokenizerlab itself, and contextlib, whose __exit__ resumes a
# generator-based context manager (e.g. the atomic directory swap) on the user's behalf.
_LIBRARY_FILES = (
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep,
    os.path.abspath(contextlib.__file__),
)


def warn_user(message: str) -> None:
    """Issue a UserWarning attributed to the first calling frame outside tokenizerlab."""
    stacklevel = 2  # the caller of warn_user
    frame = sys._getframe(1)
    while frame.f_back is not None and frame.f_code.co_filename.startswith(_LIBRARY_FILES):
        frame = frame.f_back
        stacklevel += 1
    warnings.warn(message, stacklevel=stacklevel)
