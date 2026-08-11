"""Allow ``python -m gcr`` as an alias for the installed ``gcr`` command."""

from gcr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
