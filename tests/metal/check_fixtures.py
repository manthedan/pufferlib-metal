#!/usr/bin/env python3
"""Validate external capture contracts and file sizes, without GPU work."""
import sys
from pathlib import Path


def main():
    try:
        from fixture_validation import check_fixtures
        missing = check_fixtures(Path(sys.argv[1]), sys.argv[2:])
    except Exception as error:
        # Exit 1 is reserved for missing data; checker errors must never become
        # allowable skips in run_all.sh.
        print(f'Invalid fixture: {error}', file=sys.stderr)
        return 2
    if missing:
        print(f'MISSING fixtures under {sys.argv[1]}: {len(missing)} files; first: {missing[0]}')
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
