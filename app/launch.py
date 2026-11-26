import os
import sys
from pathlib import Path
from typing import Optional, Sequence


def main(argv: Optional[Sequence[str]] = None) -> None:
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    os.environ.setdefault("STREAMLIT_SERVER_MAX_UPLOAD_SIZE", "50")
    from streamlit.web import cli as streamlit_cli

    extra = list(sys.argv[1:] if argv is None else argv)
    main_py = Path(__file__).resolve().with_name("main.py")
    sys.argv = ["streamlit", "run", str(main_py), *extra]
    sys.exit(streamlit_cli.main())


def cli() -> None:
    from app.cli import run

    run(prog="group-diversifier")


if __name__ == "__main__":
    main()
