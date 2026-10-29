"""Convenience wrapper for the validate-results CLI."""
import sys
from neuralshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["validate-results", *sys.argv[1:]]))
