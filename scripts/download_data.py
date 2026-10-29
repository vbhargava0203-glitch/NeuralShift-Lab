"""Convenience wrapper for the download-data CLI."""
import sys
from neuralshift.cli import main

if __name__ == "__main__":
    raise SystemExit(main(["download-data", *sys.argv[1:]]))
