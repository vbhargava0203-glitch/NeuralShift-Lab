"""Consistent console logging for research execution."""

import logging


def configure_logging():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    logging.getLogger("fontTools").setLevel(logging.WARNING)
