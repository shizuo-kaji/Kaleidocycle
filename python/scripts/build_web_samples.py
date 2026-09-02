"""Validate canonical samples and build the static ``file://`` fallback."""

from pathlib import Path

from kaleidocycle import build_web_sample_fallback, write_sample_catalogue


ROOT = Path(__file__).resolve().parents[1]
DATA_DIRECTORY = ROOT / "data" / "kaleidocycles"
WEB_FALLBACK = ROOT / "web" / "sample-fallback.js"


def main() -> None:
    """Refresh the catalogue and its generated browser fallback."""

    catalogue = write_sample_catalogue(DATA_DIRECTORY)
    build_web_sample_fallback(WEB_FALLBACK, directory=DATA_DIRECTORY)
    print(
        f"Wrote {WEB_FALLBACK.relative_to(ROOT)} from "
        f"{len(catalogue['samples'])} canonical samples."
    )


if __name__ == "__main__":
    main()
