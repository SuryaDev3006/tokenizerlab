"""Fail if any function in the given Python files is longer than MAX_FUNCTION_LINES.

Ruff limits statements, not lines, so this pre-commit check enforces the line limit.
Usage: python scripts/check_function_length.py FILE [FILE ...]
"""

import ast
import sys
from collections.abc import Iterator
from pathlib import Path

MAX_FUNCTION_LINES = 60

FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def too_long_functions(path: Path) -> Iterator[tuple[FunctionNode, int]]:
    """Every function in the file whose definition spans more than MAX_FUNCTION_LINES lines."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if not isinstance(node, FunctionNode):
            continue
        length = (node.end_lineno or node.lineno) - node.lineno + 1
        if length > MAX_FUNCTION_LINES:
            yield node, length


def main(paths: list[str]) -> int:
    """Report each too-long function; return 1 if there were any."""
    failures = 0
    for name in paths:
        for node, length in too_long_functions(Path(name)):
            print(f"{name}:{node.lineno}: {node.name} is {length} lines (max {MAX_FUNCTION_LINES})")
            failures += 1
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
