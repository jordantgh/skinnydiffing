"""python -m lightweight_table_diff config.yml"""
import logging
import sys

from .runner import run_config

logging.basicConfig(
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    level=logging.INFO,
)

if len(sys.argv) < 2:
    print("Usage: python -m lightweight_table_diff <config.yml>", file=sys.stderr)
    sys.exit(1)

run_config(sys.argv[1])
