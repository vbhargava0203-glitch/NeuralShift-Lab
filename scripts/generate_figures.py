"""Convenience entry point for figures."""
import sys
from neuralshift.cli import main
if __name__ == "__main__":
    raise SystemExit(main(["figures", *sys.argv[1:]]))
