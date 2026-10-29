"""Convenience wrapper for the experiment CLI."""
import sys
from neuralshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["experiment", *sys.argv[1:]]))
