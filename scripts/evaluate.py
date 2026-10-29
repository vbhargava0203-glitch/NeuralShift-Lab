"""Convenience wrapper for the evaluate CLI."""
import sys
from neuralshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["evaluate", *sys.argv[1:]]))
