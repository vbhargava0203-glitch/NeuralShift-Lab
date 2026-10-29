"""Convenience wrapper for the train CLI."""
import sys
from neuralshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["train", *sys.argv[1:]]))
