"""CLI entry point for skinnydiffing."""

import logging
import sys

from .runner import run_config


def main() -> None:
    logging.basicConfig(
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        level=logging.INFO,
    )

    if len(sys.argv) < 2:
        print("Usage: skinnydiff <config.yml>", file=sys.stderr)
        sys.exit(1)

    run_config(sys.argv[1])


if __name__ == "__main__":
    main()
