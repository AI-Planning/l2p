"""
Helper functions for L2P CLI.
"""

import sys
import difflib

BOLD = "\033[1m"
GREEN = "\033[92m"
CYAN = "\033[96m"
YELLOW = "\033[93m"
RESET = "\033[0m"

def _input_or_exit(prompt: str = "") -> str:
    value = input(prompt).strip()
    if value == "/exit":
        print("Operation cancelled.")
        sys.exit(0)
    return value

def _show_diff(original: str, modified: str):
    """Show a colored unified diff between original and modified content."""
    diff = difflib.unified_diff(
        original.splitlines(keepends=True),
        modified.splitlines(keepends=True),
        fromfile="original",
        tofile="modified",
    )
    lines = list(diff)
    if not lines:
        print(f"  {YELLOW}No changes.{RESET}")
        return

    print(f"\n  {BOLD}Changes:{RESET}")
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("---") or line.startswith("+++"):
            print(f"  {BOLD}{line}{RESET}")
        elif line.startswith("@@"):
            print(f"  {CYAN}{line}{RESET}")
        elif line.startswith("+"):
            print(f"  {GREEN}{line}{RESET}")
        elif line.startswith("-"):
            print(f"  {YELLOW}{line}{RESET}")
        else:
            print(f"  {line}")